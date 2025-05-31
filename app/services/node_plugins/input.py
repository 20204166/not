def run(inputs, state, params):
    # Input node provides a fixed value (for simulation/testing)
    return {"value": params.get("value", 0.0)}, state