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
path2 = "../em_E.pk_topk_0.7"
path3 = "../em_E_test2.pk"
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

count = 0
for graph in l1:
    rels = graph["relations"]
    for rel in rels:
        if rel[2] in complex_rels_num:
            count += 1
            break

print(count)
