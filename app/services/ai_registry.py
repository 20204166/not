# app/services/ai_registry.py

import os
import json

STORAGE_PATH = "./models"
os.makedirs(STORAGE_PATH, exist_ok=True)

def save_ai_model(model_id, data):
    path = os.path.join(STORAGE_PATH, f"{model_id}.json")
    with open(path, "w") as f:
        json.dump(data, f)

def load_ai_model(model_id):
    path = os.path.join(STORAGE_PATH, f"{model_id}.json")
    with open(path, "r") as f:
        return json.load(f)

def list_user_models():
    files = os.listdir(STORAGE_PATH)
    return [f.replace(".json", "") for f in files if f.endswith(".json") and not f.endswith("_logs.json")]
