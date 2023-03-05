import json
import torch
from copy import deepcopy
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
len_pred = len(idx2pred)

l = pickle.load(open(path, "rb"))

rel_cnt_dic = {}
for i, data in enumerate(l):
    labels = data["labels"]
    logits = data["logits"][:, 1:]
    relation_tuple = deepcopy(data["relations"])
    sub_idxs, obj_idxs, rels = relation_tuple[:, 0], relation_tuple[:, 1], relation_tuple[:, 2]
    sub_lbs, obj_lbs = labels[sub_idxs], labels[obj_idxs]
    # [[sub_lb1, obj_lb1], [sub_lb2, obj_lb2]......]
    pairs = np.stack([sub_lbs, obj_lbs], 1).tolist()
    pairs = [(idx2lb[p[0]], idx2lb[p[1]]) for p in pairs]

    # fill in rel_dic
    # rel_dic: {rel_i: {pair_j: distribution} }
    for j, (pair, r) in enumerate(zip(pairs, rels)):
        r_name = idx2pred[int(r)]

        if r_name not in rel_cnt_dic:
            rel_cnt_dic[r_name] = {}
        if pair not in rel_cnt_dic[r_name]:
            rel_cnt_dic[r_name][pair] = 0
        rel_cnt_dic[r_name][pair] += 1

importance_dic = {}
for r, pair_cnt_dic in rel_cnt_dic.items():
    for pair in pair_cnt_dic:
        cnt = pair_cnt_dic[pair]
        triplet = (pair[0], r, pair[1])
        importance_dic[triplet] = cnt / sum(pair_cnt_dic.values())

num_node_features = 10
num_edge_features = 10
sentence = []
index = 0

for triple in importance_dic:
    sentence.append(list(triple))
# for data in l:
#     labels = data["labels"]
#     relation_tuple = data["relations"]
#     sub_idxs, obj_idxs, rels = relation_tuple[:, 0], relation_tuple[:, 1], relation_tuple[:, 2]
#     sub_lbs, obj_lbs = labels[sub_idxs], labels[obj_idxs]
#     for i in range(len(relation_tuple)):
#         sentence.append(list())
#         sentence[index].append(idx2lb[sub_lbs[i]])
#         sentence[index].append(idx2pred[rels[i]])
#         sentence[index].append(idx2lb[obj_lbs[i]])
#         index = index + 1

model = Word2Vec(sentences=sentence, vector_size=num_edge_features, window=2, min_count=1, workers=4)

predfeatures = []
for item in idx2pred:
    predfeatures.append(list(model.wv[idx2pred[item]]))

g1_index = 14
g2_index = 165
graph1 = l[g1_index]
graph2 = l[g2_index]
# print(graph1)
# print(graph2)
node1 = graph1["labels"] / len_lb
node1 = np.atleast_2d(node1)
node1 = node1.T
node2 = graph2["labels"] / len_lb
node2 = np.atleast_2d(node2)
node2 = node2.T
n1 = np.array([node1.shape[0]])
n2 = np.array([node2.shape[0]])

# A1 = np.zeros((node1.shape[0], node1.shape[0]))
# A2 = np.zeros((node2.shape[0], node2.shape[0]))
# n1 = np.array([node1.shape[0]])
# n2 = np.array([node2.shape[0]])
#
# for triple in graph1["relations"]:
#     A1[triple[0]][triple[1]] = triple[2] / len_pred
#
# for triple in graph2["relations"]:
#     A2[triple[0]][triple[1]] = triple[2] / len_pred
#
# conn1, edge1 = pygm.utils.dense_to_sparse(A1)
# conn2, edge2 = pygm.utils.dense_to_sparse(A2)

conn1 = []
edge1 = []
conn2 = []
edge2 = []
for triple in graph1["relations"]:
    conn1.append([triple[0], triple[1]])
    edge1.append(predfeatures[triple[2] - 1])
for triple in graph2["relations"]:
    conn2.append([triple[0], triple[1]])
    edge2.append(predfeatures[triple[2] - 1])
conn1 = np.array(conn1)
conn2 = np.array(conn2)
edge1 = np.array(edge1)
edge2 = np.array(edge2)
# ne1 = len(edge1)
# ne2 = len(edge2)
import functools

gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=10.)  # set affinity function
K = pygm.utils.build_aff_mat(node1, edge1, conn1, node2, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

X = pygm.rrwm(K, n1, n2)
X = pygm.hungarian(X)
print(X)

X.tolist()
for (sub, obj, rel) in graph1["relations"]:
    pair = [X[sub].index(1), X[obj].index(1)]
    if pair in conn2:
        triple1 = (idx2lb[graph1["labels"][sub]], idx2pred[rel], idx2lb[graph1["labels"][obj]])
        triple2 = (
            idx2lb[graph2["labels"][pair[0]]], idx2pred[graph2["relations"][pair[0]][pair[1]]],
            idx2lb[graph2["labels"][pair[1]]])
        if importance_dic[triple1] < importance_dic[triple2]:
            l[g2_index][pair] = rel
        else:
            l[g1_index][sub][obj] = idx2pred[graph2["relations"][pair[0]][pair[1]]]
    else:
        pass

pickle.dump(l, open("em_E_test.pk", "wb"))
