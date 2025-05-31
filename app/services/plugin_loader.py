import os
import importlib

PLUGIN_DIR = "app.services.node_plugins"

def load_node_registry():
    registry = {}
    base_path = os.path.dirname(__file__)

    for fname in os.listdir(os.path.join(base_path, "node_plugins")):
        if fname.endswith(".py") and not fname.startswith("_"):
            module_name = f"{PLUGIN_DIR}.{fname[:-3]}"
            mod = importlib.import_module(module_name)
            registry[fname[:-3]] = mod.run

    return registry

NODE_REGISTRY = load_node_registry()

def get_node_handler(node_type):
    if node_type not in NODE_REGISTRY:
        raise ValueError(f"No handler for node type: {node_type}")
    return NODE_REGISTRY[node_type]