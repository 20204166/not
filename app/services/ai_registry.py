# app/services/ai_registry.py

import os
import json
from flask import Blueprint, jsonify

STORAGE_PATH = "./models"
NODE_LIBRARY_PATH = os.path.join(os.path.dirname(__file__), "node_library.json")

os.makedirs(STORAGE_PATH, exist_ok=True)

ai_registry_bp = Blueprint("ai_registry", __name__)

@ai_registry_bp.route("/nodes/library", methods=["GET"])
def get_node_library():
    if not os.path.exists(NODE_LIBRARY_PATH):
        return jsonify([]), 200
    with open(NODE_LIBRARY_PATH, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))

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

def delete_ai_model(model_id):
    path = os.path.join(STORAGE_PATH, f"{model_id}.json")
    if os.path.exists(path):
        os.remove(path)
        return True
    return False
@ai_registry_bp.route("/models", methods=["GET"])
def list_models():
    models = list_user_models()
    return jsonify(models), 200