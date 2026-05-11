import os
import logging
import threading
import json
import boto3
import watchtower
import torch
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, field_validator
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from peft import PeftModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uuid
from datetime import datetime

# --- CLOUD CONFIGURATION ---
REGION = "eu-north-1"
BASE_MODEL = "distilbert-base-uncased"
CONFIDENCE_THRESHOLD = 0.85
# Find the path where this file is living (usually /app)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Look in the same folder as main.py
ADAPTER_PATH = ROOT_DIR

API_KEYS_FILE = os.path.join(ROOT_DIR, "api_keys.json")

LABELS = ["O", "B-VENDOR", "I-VENDOR", "B-DATE", "I-DATE", "B-AMOUNT", "I-AMOUNT"]
id2label = {i: label for i, label in enumerate(LABELS)}
label2id = {label: i for i, label in enumerate(LABELS)}

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InvoiceAI")
try:
    cw_client = boto3.client("logs", region_name=REGION)
    cw_handler = watchtower.CloudWatchLogHandler(
        log_group="InvoiceAI-Backend",
        stream_name="Production-Logs",
        boto3_client=cw_client,
        send_interval=1
    )
    logger.addHandler(cw_handler)
    logger.info("--- SERVER BOOTING (Cloud Mode) ---")
except Exception as e:
    print(f"CloudWatch setup failed: {e}")

# --- API KEY AUTHENTICATION ---
def load_api_keys():
    try:
        with open(API_KEYS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data["valid_keys"])
    except Exception as e:
        logger.error(f"Failed to load API keys from {API_KEYS_FILE}: {e}")
        return set()

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="API Key required. Send 'x-api-key' in header")
    valid_keys = load_api_keys()
    if x_api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key

# --- REVIEW QUEUE STORAGE ---
# In a real app, use DynamoDB. For this demo, we use a simple list.
review_queue = []

# --- BACKGROUND MODEL LOADING ---
ml_models = {"nlp": None, "ready": False, "error": None}

def load_model_task():
    try:
        logger.info(f"STARTING MODEL LOAD: {ADAPTER_PATH}")
        if not os.path.exists(ADAPTER_PATH):
            raise Exception(f"Directory {ADAPTER_PATH} not found!")

        # 1. Load Base Model
        base_model = AutoModelForTokenClassification.from_pretrained(
            BASE_MODEL, 
            num_labels=len(LABELS),
            id2label=id2label,
            label2id=label2id
        )
        
        # 2. Load Adapters & Tokenizer
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
        
        if "distilbert" in BASE_MODEL.lower():
            tokenizer.model_input_names = ["input_ids", "attention_mask"]

        # 3. Setup Pipeline
        ml_models["nlp"] = pipeline(
            "ner", 
            model=model, 
            tokenizer=tokenizer, 
            aggregation_strategy="simple",
            device=-1 # CPU
        )
        ml_models["ready"] = True
        logger.info("SUCCESS: AI Model loaded and ready!")
        
    except Exception as e:
        logger.error(f"MODEL LOAD FAILED: {str(e)}", exc_info=True)
        ml_models["error"] = str(e)

# Start background thread
threading.Thread(target=load_model_task).start()

# --- FASTAPI APP ---
app = FastAPI(title="Invoice AI Extraction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELS ---
class ExtractionRequest(BaseModel):
    text: str
    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        if not v or not v.strip(): raise ValueError('Text cannot be empty')
        if len(v.strip()) < 5: raise ValueError('Text too short')
        return v.strip()

class Entity(BaseModel):
    type: str
    value: str
    score: float

class ExtractionResponse(BaseModel):
    id: str
    entities: List[Entity]
    raw_text: str
    status: str # "CONFIDENT", "UNCERTAIN", "VERIFIED"
    confidence_score: float
    audit_log: Optional[List[str]] = []

class VerificationRequest(BaseModel):
    extraction_id: str
    entities: List[Entity]
    reviewer: str

# --- ENDPOINTS ---

@app.get("/health")
async def health():
    if ml_models["error"]:
        return {"status": "error", "detail": ml_models["error"]}
    return {"status": "ready" if ml_models["ready"] else "loading"}

@app.post("/extract", response_model=ExtractionResponse)
async def extract_entities(request: ExtractionRequest, api_key: str = Depends(verify_api_key)):
    nlp = ml_models.get("nlp")
    if nlp is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        results = nlp(request.text)
        entities = []
        total_score = 0
        for res in results:
            etype = res.get('entity_group') or res.get('entity') or "UNKNOWN"
            if etype.startswith('B-') or etype.startswith('I-'):
                etype = etype[2:]
            score = float(res['score'])
            entities.append(Entity(
                type=etype,
                value=res['word'],
                score=score
            ))
            total_score += score
        
        avg_confidence = total_score / len(entities) if entities else 0
        status = "CONFIDENT" if avg_confidence >= CONFIDENCE_THRESHOLD else "UNCERTAIN"
        extraction_id = str(uuid.uuid4())
        
        response = ExtractionResponse(
            id=extraction_id,
            entities=entities,
            raw_text=request.text,
            status=status,
            confidence_score=avg_confidence,
            audit_log=[f"[{datetime.now().isoformat()}] System extracted with status: {status}"]
        )

        # Route to review queue if uncertain
        if status == "UNCERTAIN":
            review_queue.append(response)
            logger.warning(f"AUDIT_TRAIL: Extraction {extraction_id} routed to REVIEW_QUEUE. Confidence: {avg_confidence:.2f}")
        else:
            logger.info(f"AUDIT_TRAIL: Extraction {extraction_id} AUTO_VERIFIED. Confidence: {avg_confidence:.2f}")

        return response
    except Exception as e:
        logger.error(f"Extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/review-queue", response_model=List[ExtractionResponse])
async def get_review_queue(api_key: str = Depends(verify_api_key)):
    return review_queue

@app.post("/verify", response_model=ExtractionResponse)
async def verify_extraction(verify_req: VerificationRequest, api_key: str = Depends(verify_api_key)):
    # Find item in queue
    global review_queue
    item_index = next((i for i, item in enumerate(review_queue) if item.id == verify_req.extraction_id), None)
    
    if item_index is None:
        raise HTTPException(status_code=404, detail="Extraction not found in review queue")
    
    original_item = review_queue.pop(item_index)
    
    # Update item
    original_item.entities = verify_req.entities
    original_item.status = "VERIFIED"
    original_item.audit_log.append(f"[{datetime.now().isoformat()}] Human Verified by {verify_req.reviewer}")
    
    logger.info(f"AUDIT_TRAIL: Extraction {original_item.id} HUMAN_VERIFIED by {verify_req.reviewer}")
    
    return original_item

@app.get("/")
async def root():
    return {"message": "Invoice AI Backend is active"}
