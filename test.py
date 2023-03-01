import numpy as np
import pygmtools as pygm
import pickle
pygm.BACKEND = 'numpy'
np.random.seed(1)

path = "em_E.pk"
l = pickle.load(open(path, "rb"))

node1 = l[0]["labels"]
node2 = l[1]["labels"]
print(node1.shape[0])