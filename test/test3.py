import pickle
import numpy as np

path = "../em_E.pk"

l = pickle.load(open(path, "rb"))

graph = l[0]
rels = graph["relations"]
rel = rels[0]
a = np.array([1, 2, 3])
b = np.array([True, False, True])
print(a[b])
pass
