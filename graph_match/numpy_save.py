import numpy as np

x = [2, 3]
y = np.array([5, 6], dtype=np.float64)
for i in range(2):
    y[i] /= x[i]
print(y)
pass
