from flask import Blueprint, request, jsonify
import uuid
import os
import json

from app.services.graph_simulator import run_simulation
from app.services.ai_registry import save_ai_model, load_ai_model, list_user_models

visual_ai_bp = Blueprint('visual_ai', __name__, url_prefix='/ai')

# Store files in a local folder (or switch to DB later)
STORAGE_PATH = "./ai_models"
NODE_LIBRARY_PATH ="app/services/node_library.json"


os.makedirs(STORAGE_PATH, exist_ok=True)

@visual_ai_bp.route('/create', methods=['POST'])
def create_model():
    data = request.json
    model_id = str(uuid.uuid4())

    with open(os.path.join(STORAGE_PATH, f"{model_id}.json"), "w") as f:
        json.dump(data, f)

    return jsonify({"status": "ok", "model_id": model_id})


@visual_ai_bp.route('/train', methods=['POST'])
def train_model():
    payload = request.json
    model_id = payload["model_id"]
    dataset_name = payload["dataset"]  # Assume pre-stored datasets

    with open(os.path.join(STORAGE_PATH, f"{model_id}.json"), "r") as f:
        graph = json.load(f)

    logs = run_simulation(graph, dataset_name)
    with open(os.path.join(STORAGE_PATH, f"{model_id}_logs.json"), "w") as f:
        json.dump(logs, f)

    return jsonify({"status": "training_complete"})


@visual_ai_bp.route('/<model_id>/logs', methods=['GET'])
def get_logs(model_id):
    try:
        with open(os.path.join(STORAGE_PATH, f"{model_id}_logs.json"), "r") as f:
            logs = json.load(f)
        return jsonify(logs)
    except FileNotFoundError:
        return jsonify({"error": "Logs not found"}), 404


@visual_ai_bp.route('/list', methods=['GET'])
def list_models():
    files = os.listdir(STORAGE_PATH)
    model_files = [f for f in files if f.endswith(".json") and not f.endswith("_logs.json")]
    models = [{"model_id": f.replace(".json", "")} for f in model_files]
    return jsonify(models)


@visual_ai_bp.route('/nodes/library', methods=['GET'])
def get_node_library():
    try:
        with open(NODE_LIBRARY_PATH, "r") as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({"error": "Node library not found"}), 404
