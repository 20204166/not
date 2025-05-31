def run(inputs, state, params):
    x = sum(inp.get("value", 0.0) for inp in inputs.values())
    weight = params.get("weight", 1.0)
    bias = params.get("bias", 0.0)
    return {"value": weight * x + bias}, state