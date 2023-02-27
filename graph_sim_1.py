# compute graph similarity according to the rate of common labels
import pickle
from tqdm import tqdm

threshold = 0.5

path = "em_E.pk"
l = pickle.load(open(path, "rb"))
total = len(l) * (len(l) - 1) / 2
cnt = 0

for i, graph1 in tqdm(enumerate(l)):
    for graph2 in l[i + 1:]:
        label1 = graph1["labels"]
        label2 = graph2["labels"]
        inter = len(set(label1).intersection(set(label2)))
        rate = inter / (len(label1) + len(label2) - inter)
        if rate >= threshold:
            cnt = cnt + 1

print(cnt)
print(total)
print(cnt / total)
