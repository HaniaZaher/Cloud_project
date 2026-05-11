#preprocessing#
from datasets import load_dataset
import pandas as pd
import re
import json

dataset = load_dataset("Navneetkumar11/rvl-cdip-invoice-extracted", split="train")

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    if not text:
        return ""

    text = text.lower()
    # remove new lines
    text = text.replace("\n", " ")
    # remove unwanted words
    text = text.replace("unknown", " ")
    # remove currency patterns
    text = re.sub(r"\$[\d,]+\.?\d*", " amount ", text)
    # remove date patterns
    text = re.sub(r"\d{4}-\d{2}-\d{2}", " date ", text)
    # remove special symbols (# / \)
    text = re.sub(r"[#/\\]", " ", text)
    # keep only letters & numbers
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    # remove extra spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =========================
# CLEAN AMOUNT
# =========================
def clean_amount(x):
    try:
        if x is None:
            return None

        x = str(x)
        # remove commas
        x = x.replace(",", "")
        # keep numbers and dot only
        x = re.sub(r"[^0-9.]", "", x)
        # avoid multiple dots
        if x.count(".") > 1:
            return None

        if x == "":
            return None

        return float(x)

    except:
        return None

# =========================
# CLEAN DATE
# =========================
from datetime import datetime

def clean_date(date):
    if not date:
        return ""

    try:
        date = str(date).replace("/", "-")

        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y"):
            try:
                parsed = datetime.strptime(date, fmt)
                return parsed.strftime("%Y-%m-%d")
            except:
                pass

        return ""

    except:
        return ""

# =========================
# CLEAN VENDOR
# =========================
def clean_vendor(vendor):
    if not vendor:
        return ""

    vendor = vendor.lower()

    vendor = vendor.replace(".", " ")

    vendor = re.sub(r"[#/\\]", " ", vendor)

    vendor = vendor.replace("unknown", "")

    vendor = re.sub(r"[^a-z0-9 ]", " ", vendor)

    vendor = re.sub(r"\s+", " ", vendor)

    return vendor.strip()

# =========================
# PROCESS DATA
# =========================
processed_data = []

for item in dataset:

    raw_text = item.get("raw_ocr_text", "")
    extracted_json = item.get("extracted", "{}")

    try:
        extracted = json.loads(extracted_json)
    except:
        continue

    text = clean_text(raw_text)
    vendor = clean_vendor(extracted.get("vendor", {}).get("name", ""))
    date = clean_date(extracted.get("invoice_date", ""))
    amount = clean_amount(extracted.get("total_amount", ""))

    # =========================
    # FILTER
    # =========================

    if text == "":
        continue

    if len(text.split()) < 5:
        continue

    if vendor == "" or vendor == "unknown":
        continue

    if amount is None or amount <= 0:
        continue

    if date == "":
        continue

    processed_data.append({
        "text": text,
        "vendor": vendor,
        "date": date,
        "total_amount": amount
    })

# =========================
# DATAFRAME
# =========================
df = pd.DataFrame(processed_data)

# remove duplicates
df = df.drop_duplicates(subset=["text"])

print("Final dataset size:", len(df))
print(df.head())


# =========================
# SPLIT DATA
# =========================

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

train_size = int(0.7 * len(df))
val_size = int(0.15 * len(df))

train_df = df[:train_size]
val_df = df[train_size:train_size + val_size]
test_df = df[train_size + val_size:]

print("Train size:", len(train_df))
print("Val size:", len(val_df))
print("Test size:", len(test_df))

# =========================
# SAVE SPLITS
# =========================

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

# JSON
import json

with open("train.json", "w", encoding="utf-8") as f:
    json.dump(train_df.to_dict(orient="records"), f, indent=4, ensure_ascii=False)

with open("val.json", "w", encoding="utf-8") as f:
    json.dump(val_df.to_dict(orient="records"), f, indent=4, ensure_ascii=False)

with open("test.json", "w", encoding="utf-8") as f:
    json.dump(test_df.to_dict(orient="records"), f, indent=4, ensure_ascii=False)

print("Data split and saved successfully")

# =========================
# SAVE
# =========================
df.to_csv("clean_invoice.csv", index=False)

with open("clean_invoice.json", "w", encoding="utf-8") as f:
    json.dump(df.to_dict(orient="records"), f, indent=4, ensure_ascii=False)

print("Final Clean Data جاهزة 100%")
