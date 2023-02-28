# compute the number of graphs similar to the first graph
import pickle
from tqdm import tqdm

threshold = 0.5

path = "em_E.pk"
l = pickle.load(open(path, "rb"))
graph1 = l[2]
label1 = graph1["labels"]
len1 = len(label1)
set1 = set(label1)
total = len(l) - 1
cnt = 0

for graph2 in l[3:]:
    label2 = graph2["labels"]
    inter = len(set1.intersection(set(label2)))
    rate = inter / (len1 + len(label2) - inter)
    if rate >= threshold:
        cnt = cnt + 1

print(cnt)
print(total)
print(cnt / total)
