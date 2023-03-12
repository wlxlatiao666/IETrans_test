# compute the number of graphs similar to the first graph
import pickle
from tqdm import tqdm

threshold = 0.6
cnt_num = 100

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
        label2 = graph2["labels"]
        set2 = set(label2)
        # inter = len([x for x in label1 if x in label2])
        inter = len(set1.intersection(set2))
        rate = inter / (len1 + len(set2) - inter)
        if rate >= threshold:
            # print(l.index(graph1))
            # print(l.index(graph2))
            # print(graph1["img_path"])
            # print(graph2["img_path"])
            cnt = cnt + 1
    cnt_lst.append(cnt)

print(cnt_lst)
print(sum(cnt_lst) / (cnt_num * total))
