import json
import numpy as np
import pygmtools as pygm
import pickle
from gensim.models import Word2Vec

pygm.BACKEND = 'numpy'
np.random.seed(1)

path = "../em_E.pk"

vocab = json.load(open("../VG-SGG-dicts-with-attri.json", "r"))
idx2lb = {int(k): v for k, v in vocab["idx_to_label"].items()}
lb2idx = {k: int(v) for k, v in vocab["label_to_idx"].items()}
idx2pred = {int(k): v for k, v in vocab["idx_to_predicate"].items()}
pred2idx = {k: int(v) for k, v in vocab["predicate_to_idx"].items()}
len_lb = len(idx2lb)

l = pickle.load(open(path, "rb"))

num_node_features = 10
num_edge_features = 100
sentence = []
index = 0

for data in l:
    labels = data["labels"]
    relation_tuple = data["relations"]
    sub_idxs, obj_idxs, rels = relation_tuple[:, 0], relation_tuple[:, 1], relation_tuple[:, 2]
    sub_lbs, obj_lbs = labels[sub_idxs], labels[obj_idxs]
    for i in range(len(relation_tuple)):
        sentence.append(list())
        sentence[index].append(idx2lb[sub_lbs[i]])
        sentence[index].append(idx2pred[rels[i]])
        sentence[index].append(idx2lb[obj_lbs[i]])
        index = index + 1

model = Word2Vec(sentences=sentence, vector_size=num_edge_features, window=2, min_count=1, workers=4)

predfeatures = []
for item in idx2pred:
    predfeatures.append(model.wv[idx2pred[item]])

graph1 = l[14]
graph2 = l[165]
# print(graph1)
# print(graph2)
node1 = graph1["labels"] / len_lb
node1 = np.atleast_2d(node1)
node1 = node1.T
node2 = graph2["labels"] / len_lb
node2 = np.atleast_2d(node2)
node2 = node2.T

A1 = np.zeros((node1.shape[0], node1.shape[0], num_edge_features))
A2 = np.zeros((node2.shape[0], node2.shape[0], num_edge_features))
n1 = np.array([node1.shape[0]])
n2 = np.array([node2.shape[0]])

for triple in graph1["relations"]:
    A1[triple[0]][triple[1]] = predfeatures[triple[2] - 1]

for triple in graph2["relations"]:
    A2[triple[0]][triple[1]] = predfeatures[triple[2] - 1]

conn1, edge1 = pygm.utils.dense_to_sparse(A1)
conn2, edge2 = pygm.utils.dense_to_sparse(A2)
import functools

gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.)  # set affinity function
K = pygm.utils.build_aff_mat(node1, edge1, conn1, node2, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

X = pygm.rrwm(K, n1, n2)
X = pygm.sinkhorn(X)
print(X)
pass
