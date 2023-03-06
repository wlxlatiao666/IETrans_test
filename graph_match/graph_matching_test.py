# test graph matching
import json
import torch
import numpy as np
from copy import deepcopy
import pickle
from tqdm import tqdm
import pygmtools as pygm
from gensim.models import Word2Vec
import functools

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
threshold = 0.5

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

print(rel_cnt_dic['has'][('boy', 'shirt')])

importance_dic = {}
all_triplets = []
for r, pair_cnt_dic in rel_cnt_dic.items():
    for pair in pair_cnt_dic:
        cnt = pair_cnt_dic[pair]
        triplet = (pair[0], r, pair[1])
        importance_dic[triplet] = cnt / sum(pair_cnt_dic.values())

for triple in importance_dic:
    all_triplets.append(triple)

num_node_features = 10
num_edge_features = 10
sentence = []

for triple in importance_dic:
    sentence.append(list(triple))

model = Word2Vec(sentences=sentence, vector_size=num_edge_features, window=2, min_count=1, workers=4)

predfeatures = []
for item in idx2pred:
    predfeatures.append(list(model.wv[idx2pred[item]]))


def sim_graphs(g1, g2, thres):
    label1 = g1["labels"]
    set1 = set(label1)
    label2 = g2["labels"]
    set2 = set(label2)
    inter = len(set1.intersection(set2))
    rate = inter / (len(set1) + len(set2) - inter)
    if rate >= thres:
        return True
    return False


def match_graphs(g1, g2):
    node1 = g1["labels"] / len_lb
    node1 = np.atleast_2d(node1)
    node1 = node1.T
    node2 = g2["labels"] / len_lb
    node2 = np.atleast_2d(node2)
    node2 = node2.T
    n1 = np.array([node1.shape[0]])
    n2 = np.array([node2.shape[0]])

    conn1 = g1["relations"][:, :2]
    edge1 = []
    conn2 = g2["relations"][:, :2]
    edge2 = []
    for triple in g1["relations"]:
        edge1.append(predfeatures[triple[2] - 1])
    for triple in g2["relations"]:
        edge2.append(predfeatures[triple[2] - 1])
    conn1 = np.array(conn1)
    conn2 = np.array(conn2)
    edge1 = np.array(edge1)
    edge2 = np.array(edge2)

    gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=num_edge_features)  # set affinity function
    K = pygm.utils.build_aff_mat(node1, edge1, conn1, node2, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

    X = pygm.rrwm(K, n1, n2)
    X = pygm.hungarian(X)
    return X


def fix_relations(g1, g2, g1_index, g2_index, match):
    match = match.tolist()
    label1 = g1["labels"]
    label2 = g2["labels"]
    rel1 = g1["relations"]
    rel2 = g2["relations"]

    for index1, (sub, obj, rel) in enumerate(rel1):
        if 1 not in match[sub] or 1 not in match[obj]:
            continue
        pair = [match[sub].index(1), match[obj].index(1)]
        if pair in rel2[:, :2].tolist():
            g2pair_index = rel2[:, :2].tolist().index(pair)
            triple1 = (idx2lb[label1[sub]], idx2pred[rel], idx2lb[label1[obj]])
            # print(idx2lb[g2["labels"][pair[0]]])
            # print(idx2pred[g2["relations"][g2pair_index][2]])
            triple2 = (
                idx2lb[label2[pair[0]]], idx2pred[rel2[g2pair_index][2]],
                idx2lb[label2[pair[1]]])
            if importance_dic[triple1] < importance_dic[triple2]:
                if (triple2[0], triple1[1], triple2[2]) in all_triplets:
                    l[g2_index]["relations"][g2pair_index][2] = rel
            else:
                if (triple1[0], triple2[1], triple1[2]) in all_triplets:
                    l[g1_index]["relations"][index1][2] = rel2[g2pair_index][2]
        else:
            pass


for i, graph1 in tqdm(enumerate(l)):
    for j, graph2 in enumerate(l[i + 1:]):
        if sim_graphs(graph1, graph2, threshold):
            matching_result = match_graphs(graph1, graph2)
            fix_relations(graph1, graph2, i, i + j + 1, matching_result)

pickle.dump(l, open("em_E_test.pk", "wb"))
