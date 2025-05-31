import numpy as np

def run(inputs, state, params):
    # Simplified LSTM logic
    x_t = list(inputs.values())[0].get("value", 0.0)
    h_prev = state.get("h_t", 0.0)
    c_prev = state.get("c_t", 0.0)

    f_t, i_t, o_t = 0.8, 0.7, 0.6  # dummy gates

    c_t = f_t * c_prev + i_t * x_t
    h_t = o_t * np.tanh(c_t)

    return {"value": h_t}, {"h_t": h_t, "c_t": c_t}