import numpy as np
from app.services.plugin_loader import get_node_handler
from collections import defaultdict, deque

def topological_sort(nodes, edges):
    in_degree = {nid: 0 for nid in nodes}
    adj = defaultdict(list)

    for edge in edges:
        adj[edge["from"]].append(edge["to"])
        in_degree[edge["to"]] += 1

    queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
    ordered = []

    while queue:
        node = queue.popleft()
        ordered.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(ordered) != len(nodes):
        raise ValueError("Graph has cycles or disconnected nodes")

    return ordered

def run_simulation(graph, dataset_name, timesteps=10, autosave=False, preset_name=None):
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]
    execution_order = topological_sort(nodes, edges)

    # Init state and logs
    state = {nid: {"mem": {}, "outputs": {}} for nid in nodes}
    logs = []

    # Simulate T timesteps
    for t in range(timesteps):
        timestep_log = {"timestep": t, "node_logs": {}}

        for node_id in execution_order:
            node = nodes[node_id]
            handler = get_node_handler(node["type"])

            # Gather inputs from previous outputs
            inputs = {}
            for edge in edges:
                if edge["to"] == node_id:
                    from_node = edge["from"]
                    from_port = edge.get("from_port", "default")
                    to_port = edge.get("to_port", from_node)
                    val = state[from_node]["outputs"].get(from_port, None)
                    inputs[to_port] = val

            # Run node logic
            prev_state = state[node_id]["mem"]
            params = node.get("params", {})

            try:
                outputs, new_mem = handler(inputs, prev_state, params)
                success = True
            except Exception as e:
                outputs, new_mem = {"error": str(e)}, prev_state
                success = False

            # Save new state and output
            state[node_id]["mem"] = new_mem
            state[node_id]["outputs"] = outputs

            # Log it
            timestep_log["node_logs"][node_id] = {
                "inputs": inputs,
                "outputs": outputs,
                "state": new_mem,
                "success": success
            }

        logs.append(timestep_log)

    return {
        "graph_id": graph.get("id"),
        "dataset_name": dataset_name,
        "timesteps": timesteps,
        "logs": logs
    }
