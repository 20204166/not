import os
import json
from datetime import datetime
import hashlib

# Directory to store presets
PRESET_DIR = "app/presets"

# Ensure the directory exists
os.makedirs(PRESET_DIR, exist_ok=True)

def generate_preset_id(graph):
    """
    Generate a unique hash ID based on the graph content.
    Useful when no 'id' field is provided.
    """
    raw = json.dumps(graph, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]

def save_preset(graph, simulation_result, preset_name=None):
    """
    Save the graph and its simulation result to a versioned JSON file.
    """
    graph_id = graph.get("id") or generate_preset_id(graph)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    filename = f"{preset_name or graph_id}_{timestamp}.json"
    path = os.path.join(PRESET_DIR, filename)

    preset_data = {
        "graph": graph,
        "result": simulation_result,
        "graph_id": graph_id,
        "timestamp": timestamp,
        "preset_name": preset_name or "untitled"
    }

    with open(path, "w") as f:
        json.dump(preset_data, f, indent=2)

    print(f"[✔] Preset saved: {filename}")
    return path

def load_preset(filepath):
    """
    Load a preset JSON file from disk.
    """
    with open(filepath, "r") as f:
        return json.load(f)

def list_presets():
    """
    List all saved preset files in the preset directory.
    """
    return [f for f in os.listdir(PRESET_DIR) if f.endswith(".json")]
