from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, field_validator
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from peft import PeftModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import json  # New

# Configuration
BASE_MODEL = "distilbert-base-uncased"
ADAPTER_PATH = r"C:\Users\nohaa\OneDrive\Desktop\cloud" 
LABELS = ["O", "B-VENDOR", "I-VENDOR", "B-DATE", "I-DATE", "B-AMOUNT", "I-AMOUNT"]
id2label = {i: label for i, label in enumerate(LABELS)}
label2id = {label: i for i, label in enumerate(LABELS)}

# ============  Load API Keys ============
API_KEYS_FILE = r"C:\Users\nohaa\OneDrive\Desktop\cloud\backend\api_keys.json"  #api_keys.json file path 

def load_api_keys():
    with open(API_KEYS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data["valid_keys"])

# ============ Authentication ============
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="API Key required. Send 'x-api-key' in header")
    
    valid_keys = load_api_keys()
    
    if x_api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    
    return x_api_key

# Global container for model and tokenizer
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    print("Loading base model...")
    base_model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL, 
        num_labels=len(LABELS), 
        id2label=id2label, 
        label2id=label2id
    )
    
    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    
    # Handle model-specific quirks
    if "distilbert" in BASE_MODEL.lower():
        tokenizer.model_input_names = ["input_ids", "attention_mask"]
    
    print("Setting up pipeline...")
    ml_models["nlp"] = pipeline(
        "ner", 
        model=model, 
        tokenizer=tokenizer, 
        aggregation_strategy="simple"
    )
    print("AI Model loaded successfully.")
    yield
    # Clean up ML models and release resources
    ml_models.clear()

app = FastAPI(title="Invoice AI Extraction API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ Models with Validation ============
class ExtractionRequest(BaseModel):
    text: str

    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        if len(v.strip()) < 10:
            raise ValueError('Text must be at least 5 characters')
        if len(v.strip()) > 5000:
            raise ValueError('Text cannot exceed 5000 characters')
        return v.strip()

class Entity(BaseModel):
    type: str
    value: str
    score: float

class ExtractionResponse(BaseModel):
    entities: List[Entity]
    raw_text: str

# ============ Endpoints ============

@app.get("/health")
async def health_check():
    return {"status": "ready" if ml_models.get("nlp") else "loading"}

# ============ Protected Endpoint ============
@app.post("/extract", response_model=ExtractionResponse)
async def extract_entities(
    request: ExtractionRequest,
    api_key: str = Depends(verify_api_key)  # New: API key verification
):
    nlp = ml_models.get("nlp")
    if nlp is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        print(f"Analyzing text: {request.text[:50]}...")
        results = nlp(request.text)
        print(f"AI Results: {results}")
        
        entities = []
        for res in results:
            etype = res.get('entity_group') or res.get('entity') or "UNKNOWN"
            if etype.startswith('B-') or etype.startswith('I-'):
                etype = etype[2:]
                
            entities.append(Entity(
                type=etype,
                value=res['word'],
                score=float(res['score'])
            ))
        return ExtractionResponse(entities=entities, raw_text=request.text)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
