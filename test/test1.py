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


def to_dic(data):
    dic = {}
    for d in data:
        dic[d['img_path']] = d
    return dic


path1 = "../em_E.pk"
path2 = "../graph_match/em_E_5.pk"
path3 = "../em_E.pk_topk_0.7"
path4 = "../em_E1.pk"

vocab = json.load(open("../VG-SGG-dicts-with-attri.json", "r"))
idx2lb = {int(k): v for k, v in vocab["idx_to_label"].items()}
lb2idx = {k: int(v) for k, v in vocab["label_to_idx"].items()}
idx2pred = {int(k): v for k, v in vocab["idx_to_predicate"].items()}
pred2idx = {k: int(v) for k, v in vocab["predicate_to_idx"].items()}
predcnt = {k: int(v) for k, v in vocab["predicate_count"].items()}
predcnt = sorted(predcnt.items(), key=lambda x: x[1], reverse=True)

freq_rels_num = [31, 20, 22, 30, 48, 29, 50, 1, 21, 8, 43]
all_rels_num = [i + 1 for i in range(50)]
complex_rels_num = [i for i in all_rels_num if i not in freq_rels_num]
freq_rels = [idx2pred[i] for i in freq_rels_num]

l1 = pickle.load(open(path1, "rb"))
l2 = pickle.load(open(path2, "rb"))
l3 = pickle.load(open(path3, "rb"))
l4 = pickle.load(open(path4, "rb"))
l4 = to_dic(l4)

for i in range(len(l1)):
    path = l1[i]["img_path"]
    g4_data = l4.get(path, None)
    r1 = l1[i]["relations"]
    r2 = l2[i]["relations"]
    r3 = l3[i]["relations"]
    r4 = g4_data["relations"]
    label1 = l1[i]["labels"]
    label2 = l2[i]["labels"]
    label3 = l3[i]["labels"]
    label4 = g4_data["labels"]
    triple1 = []
    triple2 = []
    triple3 = []
    triple4 = []
    for item in r1:
        triple1.append((idx2lb[label1[item[0]]], idx2pred[item[2]], idx2lb[label1[item[1]]]))
    for item in r2:
        triple2.append((idx2lb[label2[item[0]]], idx2pred[item[2]], idx2lb[label2[item[1]]]))
    for item in r3:
        triple3.append((idx2lb[label3[item[0]]], idx2pred[item[2]], idx2lb[label3[item[1]]]))
    for item in r4:
        triple4.append((idx2lb[label4[item[0]]], idx2pred[item[2]], idx2lb[label4[item[1]]]))
    pass
