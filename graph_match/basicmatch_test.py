import json
import numpy as np
import pygmtools as pygm
import pickle

pygm.BACKEND = 'numpy'
np.random.seed(1)

path = "../em_E.pk"
l = pickle.load(open(path, "rb"))
vocab = json.load(open("../VG-SGG-dicts-with-attri.json", "r"))
idx2lb = {int(k): v for k, v in vocab["idx_to_label"].items()}
idx2pred = {int(k): v for k, v in vocab["idx_to_predicate"].items()}
len_lb = len(idx2lb)
len_pred = len(idx2pred)

graph1 = l[14]
graph2 = l[165]
print(graph1)
print(graph2)
node1 = graph1["labels"] / len_lb
node1 = np.atleast_2d(node1)
node1 = node1.T
node2 = graph2["labels"] / len_lb
node2 = np.atleast_2d(node2)
node2 = node2.T
print(node1.shape[0])

A1 = np.zeros((node1.shape[0], node1.shape[0]))
A2 = np.zeros((node2.shape[0], node2.shape[0]))
n1 = np.array([node1.shape[0]])
n2 = np.array([node2.shape[0]])

for triple in graph1["relations"]:
    A1[triple[0]][triple[1]] = triple[2] / len_pred

for triple in graph2["relations"]:
    A2[triple[0]][triple[1]] = triple[2] / len_pred

conn1, edge1 = pygm.utils.dense_to_sparse(A1)
conn2, edge2 = pygm.utils.dense_to_sparse(A2)
import functools

gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.)  # set affinity function
K = pygm.utils.build_aff_mat(node1, edge1, conn1, node2, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

X = pygm.rrwm(K, n1, n2)
X = pygm.sinkhorn(X)
print(X)
pass
