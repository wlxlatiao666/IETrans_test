# compute the number of graphs similar to the first graph
import pickle
from tqdm import tqdm

freq_rels_num = [31, 20, 22, 30, 48, 29, 50, 1, 21, 8, 43]
all_rels_num = [i + 1 for i in range(50)]
complex_rels_num = [i for i in all_rels_num if i not in freq_rels_num]


def is_complex(graph):
    rels = graph["relations"]
    for rel in rels:
        if rel[2] in complex_rels_num:
            return True

    return False

def sim_graphs(g1, g2):
    lab1 = g1["labels"]
    lab2 = g2["labels"]
    rels1 = g1["relations"]
    rels2 = g2["relations"]

    for rel1 in rels1:
        for rel2 in rels2:
            if lab1[rel1[0]] == lab2[rel2[0]] and lab1[rel1[1]] == lab2[rel2[1]] and rel2[2] in complex_rels_num:
                return True

    return False


# threshold = 0.7
cnt_num = 100
cnt_zero = 0

path = "../em_E.pk"
l = pickle.load(open(path, "rb"))
total = len(l) - 1
cnt_lst = []
# print(l[18]["labels"])
# print(l[18]["img_path"])
for i in tqdm(range(cnt_num)):
    graph1 = l[i]
    label1 = graph1["labels"]
    set1 = set(label1)
    # len1 = len(label1)
    len1 = len(set1)
    cnt = 0
    for graph2 in l[:i] + l[i + 1:]:
        # label2 = graph2["labels"]
        # set2 = set(label2)
        # inter = len([x for x in label1 if x in label2])
        # inter = len(set1.intersection(set2))
        # rate = inter / (len1 + len(set2) - inter)
        if sim_graphs(graph1, graph2):
            cnt = cnt + 1
    cnt_lst.append(cnt)
    if cnt == 0:
        cnt_zero += 1

print(cnt_lst)
print(cnt_zero)
print(sum(cnt_lst) / (cnt_num * total))
