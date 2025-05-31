def run(inputs, state, params):
    return {"value": params.get("value", 1)}, state

def metadata():
    return {
        "label": "Input Node",
        "category": "developer",
        "description": "Provides a fixed numeric input",
        "params": {
            "value": {"type": "float", "default": 1.0}
        }
    }
