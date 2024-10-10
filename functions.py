# Activation functions

def step_function(z: float) -> int:
    return 0 if z < 0 else 1


def relu(z: float) -> float:
    return 0 if z < 0 else z
