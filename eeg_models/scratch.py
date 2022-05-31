import matplotlib.pyplot as plt
import numpy as np


x = np.array([2 * a for a in range(20)])
y = np.array([b**2 for b in x])

plt.figure(figsize=(10, 10))
plt.title("Ilkinji plot")
plt.xlabel("Epochs")
plt.ylabel("(val) accuracy")
plt.plot(4 * x, y, label="accuracy")
plt.legend()
plt.show()
