# compute the number of graphs similar to the first graph
import pickle
from tqdm import tqdm
import json

vocab = json.load(open("../VG-SGG-dicts-with-attri.json", "r"))
idx2lb = {int(k): v for k, v in vocab["idx_to_label"].items()}
idx2pred = {int(k): v for k, v in vocab["idx_to_predicate"].items()}

freq_rels_num = [31, 20, 22, 30, 48, 29, 50, 1, 21, 8, 43]
all_rels_num = [i + 1 for i in range(50)]
complex_rels_num = [i for i in all_rels_num if i not in freq_rels_num]


def is_complex(graph):
    rels = graph["relations"]
    for rel in rels:
        if rel[2] in complex_rels_num:
            return True

    return False


threshold = 0.5
cnt_num = 100
cnt_zero = 0

path = "../em_E.pk"
l = pickle.load(open(path, "rb"))
total = len(l) - 1
cnt1_lst = []
cnt2_lst = []
# print(l[18]["labels"])
# print(l[18]["img_path"])
for i in tqdm(range(cnt_num)):
    graph1 = l[i]
    label1 = graph1["labels"]
    set1 = set(label1)
    # len1 = len(label1)
    len1 = len(set1)
    cnt1 = 0
    cnt2 = 0
    for graph2 in l[:i] + l[i + 1:]:
        label2 = graph2["labels"]
        set2 = set(label2)
        # inter = len([x for x in label1 if x in label2])
        inter = len(set1.intersection(set2))
        rate = inter / (len1 + len(set2) - inter)
        if rate >= threshold and is_complex(graph2):
            # print(l.index(graph1))
            # print(l.index(graph2))
            # print(graph1["img_path"])
            # print(graph2["img_path"])
            cnt1 = cnt1 + 1
        if inter > 0:
            indexes = []
            # for index, label in enumerate(label2):
            #     if label in label1:
            #         indexes.append(index)
            for index, rel in enumerate(graph2["relations"]):
                if label2[rel[0]] in label1 and label2[rel[1]] in label1 and rel[2] in complex_rels_num:
                    indexes.append(index)
                    triple = (idx2lb[label2[rel[0]]], idx2pred[rel[2]], idx2lb[label2[rel[1]]])
                    cnt2 = cnt2 + 1
                    # break
            # if len(indexes) != 0:
            #     print(len(indexes))
    cnt1_lst.append(cnt1)
    cnt2_lst.append(cnt2)
    if cnt1 == 0:
        cnt_zero += 1

print(cnt1_lst)
print(cnt2_lst)
print(cnt_zero)
print(sum(cnt1_lst) / (cnt_num * total))
print(sum(cnt2_lst) / (cnt_num * total))
