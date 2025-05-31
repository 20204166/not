import os
import importlib

# Location of the plugin module (should match your package structure)
PLUGIN_DIR = "app.services.node_plugins"

def load_node_registry():
    registry = {}
    metadata_registry = {}
    base_path = os.path.dirname(__file__)

    plugin_path = os.path.join(base_path, "node_plugins")

    for fname in os.listdir(plugin_path):
        if fname.endswith(".py") and not fname.startswith("_"):
            module_name = f"{PLUGIN_DIR}.{fname[:-3]}"
            try:
                mod = importlib.import_module(module_name)

                # Register run() function
                if hasattr(mod, "run"):
                    registry[fname[:-3]] = mod.run
                else:
                    print(f"[Warning] Plugin {fname} is missing 'run()' function.")

                # Register metadata if available
                if hasattr(mod, "metadata"):
                    metadata_registry[fname[:-3]] = mod.metadata()

            except Exception as e:
                print(f"[Error] Failed to load plugin {fname}: {e}")

    return registry, metadata_registry

# Two registries: one for running nodes, one for metadata
NODE_REGISTRY, NODE_METADATA = load_node_registry()

def get_node_handler(node_type):
    if node_type not in NODE_REGISTRY:
        raise ValueError(f"No handler for node type: {node_type}")
    return NODE_REGISTRY[node_type]

def get_node_metadata(node_type):
    return NODE_METADATA.get(node_type, {})
def list_node_types():
    """List all available node types."""
    return list(NODE_REGISTRY.keys())