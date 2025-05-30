
import numpy as np
from app.services.node_registry import get_node_handler

def run_simulation(graph, dataset_name, timesteps=10):
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]

    # Build adjacency list for execution order
    forward_map = {nid: [] for nid in nodes}
    for edge in edges:
        forward_map[edge["from"]].append(edge["to"])

    # Init state and logs
    state = {nid: {"mem": {}, "outputs": {}} for nid in nodes}
    logs = []

    # Simulate T timesteps
    for t in range(timesteps):
        timestep_log = {"timestep": t, "node_logs": {}}

        for node_id, node in nodes.items():
            handler = get_node_handler(node["type"])

            # Gather inputs from previous outputs
            inputs = {}
            for edge in edges:
                if edge["to"] == node_id:
                    from_node = edge["from"]
                    inputs[from_node] = state[from_node]["outputs"]

            # Run node logic
            prev_state = state[node_id]["mem"]
            params = node.get("params", {})
            outputs, new_mem = handler(inputs, prev_state, params)

            # Save new state and output
            state[node_id]["mem"] = new_mem
            state[node_id]["outputs"] = outputs

            # Log it
            timestep_log["node_logs"][node_id] = {
                "inputs": inputs,
                "outputs": outputs,
                "state": new_mem
            }

        logs.append(timestep_log)

    return logs
