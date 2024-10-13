import matplotlib.pyplot as plt
import numpy as np

def step(z: float) -> int:
    return 0 if z < 0 else 1

def sigmoid(z: float) -> float:
    return 1 / (1 + np.exp(-z))


def sigmoid_der(z: float) -> float:
    return sigmoid(z) * (1 - sigmoid(z))


x = np.linspace(-20, 20)
y = np.heaviside(x, 1)
# y = [sigmoid(_) for _ in x]
# y1 = [sigmoid_der(_) for _ in x]

plt.step(x,y)
plt.title("Step function")
plt.ylabel("f(x)")
plt.xlabel("x")

# plt.plot(x, y)
# plt.plot(x, y1)
plt.show()