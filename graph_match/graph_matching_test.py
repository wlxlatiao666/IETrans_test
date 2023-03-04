# test graph matching
import numpy as np
import pickle
from tqdm import tqdm
import pygmtools as pygm


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
    node1 = g1["labels"]
    node2 = g2["labels"]
    A1 = np.zeros((node1.shape[0], node1.shape[0]))
    A2 = np.zeros((node2.shape[0], node2.shape[0]))

    pass


pygm.BACKEND = 'numpy'
np.random.seed(1)
threshold = 0.5

path = "../em_E.pk"
l = pickle.load(open(path, "rb"))
graph11 = l[0]["labels"]
for i, graph1 in tqdm(enumerate(l)):
    for graph2 in l[i + 1:]:
        if sim_graphs(graph1, graph2, threshold):
            matching_result = match_graphs(graph1, graph2)
