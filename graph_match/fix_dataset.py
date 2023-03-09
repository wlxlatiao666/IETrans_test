import pickle
from tqdm import tqdm
import numpy as np

path = "../em_E.pk"
l = pickle.load(open(path, "rb"))
# i = 4784
# graph = l[i]
# print(type(graph['boxes']))
for i, graph in tqdm(enumerate(l)):
    del graph['logits']
    del_label_list = []
    del_rel_list = []
    boxes = graph['boxes']
    labels = graph['labels']
    relations = graph['relations']

    for j, box in enumerate(boxes):
        if box[0] == box[2] or box[1] == box[3]:
            del_label_list.append(j)
            for k, rel in enumerate(relations):
                if rel[0] == j or rel[1] == j:
                    del_rel_list.append(k)
    del_rel_list = set(del_rel_list)
    del_rel_list = list(del_rel_list)

    del_label_list.reverse()
    del_rel_list.sort(reverse=True)
    for label in del_label_list:
        # del l[i]['boxes'][label]
        boxes = np.delete(boxes, label, axis=0)
        # del l[i]['labels'][label]
        labels = np.delete(labels, label, axis=0)
    for rel in del_rel_list:
        # del l[i]['relations'][rel]
        relations = np.delete(relations, rel, axis=0)
    for j, rel in enumerate(relations):
        minus0 = 0
        minus1 = 0
        for num in del_label_list:
            if num < rel[0]:
                minus0 += 1
            if num < rel[1]:
                minus1 += 1
        relations[j][0] -= minus0
        relations[j][1] -= minus1

    l[i]['boxes'] = boxes
    l[i]['labels'] = labels
    l[i]['relations'] = relations

# print(l[i])
pickle.dump(l, open("em_E_fixed.pk", "wb"))
