import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable
import torch.cuda
import torchvision.transforms as transforms
import cv2
from PIL import Image

pygm.BACKEND = 'numpy'
np.random.seed(1)

path = "../em_E.pk"

vocab = json.load(open("../VG-SGG-dicts-with-attri.json", "r"))
idx2lb = {int(k): v for k, v in vocab["idx_to_label"].items()}
lb2idx = {k: int(v) for k, v in vocab["label_to_idx"].items()}
idx2pred = {int(k): v for k, v in vocab["idx_to_predicate"].items()}
pred2idx = {k: int(v) for k, v in vocab["predicate_to_idx"].items()}
len_lb = len(idx2lb)
# len_pred = len(idx2pred)

l = pickle.load(open(path, "rb"))
threshold = 0.5
num_graphs = 50

rel_cnt_dic = {}
for i, data in enumerate(l):
    labels = data["labels"]
    # logits = data["logits"][:, 1:]
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
all_triplets = []
for r, pair_cnt_dic in rel_cnt_dic.items():
    for pair in pair_cnt_dic:
        cnt = pair_cnt_dic[pair]
        triplet = (pair[0], r, pair[1])
        importance_dic[triplet] = cnt / sum(pair_cnt_dic.values())

for triple in importance_dic:
    all_triplets.append(triple)

num_features = 512
TARGET_IMG_SIZE = 224
img_to_tensor = transforms.ToTensor()
sentence = []

for triple in importance_dic:
    sentence.append(list(triple))

model = Word2Vec(sentences=sentence, vector_size=num_features, window=2, min_count=1, workers=4)

labelfeatures = np.load('lb_feature.npy')
predfeatures = []
for item in idx2pred:
    predfeatures.append(list(model.wv[idx2pred[item]]))

freq_rels_num = [31, 20, 22, 30, 48, 29, 50, 1, 21, 8, 43]
all_rels_num = [i + 1 for i in range(50)]
complex_rels_num = [i for i in all_rels_num if i not in freq_rels_num]


def is_complex(graph):
    rels = graph["relations"]
    for rel in rels:
        if rel[2] in complex_rels_num:
            return True

    return False


def make_model():
    model = models.vgg16(pretrained=True).features[:28]  # 其实就是定位到第28层，对照着上面的key看就可以理解
    model = model.eval()  # 一定要有这行，不然运算速度会变慢（要求梯度）而且会影响结果
    model.cuda()  # 将模型从CPU发送到GPU,如果没有GPU则删除该行
    return model


model = make_model()


# 特征提取
def extract_feature(model, img):
    model.eval()

    img = cv2.resize(img, (TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    tensor = img_to_tensor(img)
    tensor = tensor.cuda()

    result = model(Variable(tensor))
    pool = torch.nn.MaxPool2d(kernel_size=14, stride=14)
    result = pool(result)
    result = torch.flatten(result)
    result_npy = result.data.cpu().numpy()

    return result_npy


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
    node1 = []
    for label in g1['labels']:
        node1.append(labelfeatures[label - 1])
    node1 = np.array(node1)
    node2 = []
    for label in g2['labels']:
        node2.append(labelfeatures[label - 1])
    node2 = np.array(node2)

    # node1 = g1["labels"] / len_lb
    # node1 = np.atleast_2d(node1)
    # node1 = node1.T
    # node2 = g2["labels"] / len_lb
    # node2 = np.atleast_2d(node2)
    # node2 = node2.T

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

    # if edge1.shape[0] == 0:
    #     edge1 = edge1[:, None]
    # if edge2.shape[0] == 0:
    #     edge2 = edge2[:, None]

    gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=num_features)  # set affinity function
    K = pygm.utils.build_aff_mat(node1, edge1, conn1, node2, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

    X = pygm.rrwm(K, n1, n2)
    X = pygm.hungarian(X)
    return X


# def fix_relations(g1, g2, g1_index, g2_index, match):
#     match = match.tolist()
#     label1 = g1["labels"]
#     label2 = g2["labels"]
#     rel1 = g1["relations"]
#     rel2 = g2["relations"]
#
#     for index1, (sub, obj, rel) in enumerate(rel1):
#         if 1 not in match[sub] or 1 not in match[obj]:
#             continue
#         pair = [match[sub].index(1), match[obj].index(1)]
#         if pair in rel2[:, :2].tolist():
#             g2pair_index = rel2[:, :2].tolist().index(pair)
#             triple1 = (idx2lb[label1[sub]], idx2pred[rel], idx2lb[label1[obj]])
#             triple2 = (
#                 idx2lb[label2[pair[0]]], idx2pred[rel2[g2pair_index][2]],
#                 idx2lb[label2[pair[1]]])
#             if importance_dic[triple1] < importance_dic[triple2]:
#                 if (triple2[0], triple1[1], triple2[2]) in all_triplets:
#                     l[g2_index]["relations"][g2pair_index][2] = rel
#             else:
#                 if (triple1[0], triple2[1], triple1[2]) in all_triplets:
#                     l[g1_index]["relations"][index1][2] = rel2[g2pair_index][2]
#         else:
#             pass

def overlap(box1, box2):
    if box1[0] > box2[2] or box2[0] > box1[2]:
        return False
    if box1[1] > box2[3] or box2[1] > box1[3]:
        return False
    return True


def fix_relations(g1, g2, g1_index, g2_index, match):
    match = match.tolist()
    label1 = g1["labels"]
    label2 = g2["labels"]
    boxes1 = g1["boxes"]
    boxes2 = g2["boxes"]
    relations1 = g1["relations"]
    relations2 = g2["relations"]

    # ITrans
    for index1, (sub1, obj1, rel1) in enumerate(relations1):
        for index2, (sub2, obj2, rel2) in enumerate(relations2):
            if match[sub1][sub2] == 1 and match[obj1][obj2] == 1:
                triple1 = (idx2lb[label1[sub1]], idx2pred[rel1], idx2lb[label1[obj1]])
                triple2 = (idx2lb[label2[sub2]], idx2pred[rel2], idx2lb[label2[obj2]])
                new_triple1 = (triple1[0], triple2[1], triple1[2])
                new_triple2 = (triple2[0], triple1[1], triple2[2])
                if new_triple1 in all_triplets and importance_dic[new_triple1] > importance_dic[triple1]:
                    l[g1_index]["relations"][index1][2] = rel2
                if new_triple2 in all_triplets and importance_dic[new_triple2] > importance_dic[triple2]:
                    l[g2_index]["relations"][index2][2] = rel1

    # ETrans
    for index1, (sub1, obj1, rel1) in enumerate(relations1):
        if 1 not in match[sub1] or 1 not in match[obj1]:
            continue
        pair = [match[sub1].index(1), match[obj1].index(1)]
        if pair in relations2[:, :2].tolist():
            continue
        if not overlap(boxes2[pair[0]], boxes2[pair[1]]):
            continue
        triple = (idx2lb[label2[pair[0]]], idx2pred[rel1], idx2lb[label2[pair[1]]])
        if triple in all_triplets and rel1 in complex_rels_num:
            l[g2_index]["relations"] = np.row_stack((l[g2_index]["relations"], [pair[0], pair[1], rel1]))

    match = np.array(match).T
    match = match.tolist()
    for index2, (sub2, obj2, rel2) in enumerate(relations2):
        if 1 not in match[sub2] or 1 not in match[obj2]:
            continue
        pair = [match[sub2].index(1), match[obj2].index(1)]
        if pair in relations1[:, :2].tolist():
            continue
        if not overlap(boxes1[pair[0]], boxes1[pair[1]]):
            continue
        triple = (idx2lb[label1[pair[0]]], idx2pred[rel2], idx2lb[label1[pair[1]]])
        if triple in all_triplets and rel2 in complex_rels_num:
            l[g1_index]["relations"] = np.row_stack((l[g1_index]["relations"], [pair[0], pair[1], rel2]))

    return


# len_intra_data = len(l)
# similarity = [[[0] * len_intra_data] * len_intra_data]
# l = l[598:]

for i in tqdm(range(len(l))):
    num_matched_graphs = 0
    for j in range(i + 1, len(l)):
        if sim_graphs(l[i], l[j], threshold) and is_complex(l[j]):
            matching_result = match_graphs(l[i], l[j])
            fix_relations(l[i], l[j], i, j, matching_result)
            num_matched_graphs += 1
            if num_matched_graphs >= num_graphs:
                break
    # print(i)
    # if i == 100:
    #     break

pickle.dump(l, open("em_E_100.pk", "wb"))
