import pickle
import json

complex_relation = []
freq_rels_num = [31, 20, 22, 30, 48, 29, 50, 1, 21, 8, 43, 40, 49, 41, 23]
vocab = json.load(open("../VG-SGG-dicts-with-attri.json", "r"))
idx2pred = {int(k): v for k, v in vocab["idx_to_predicate"].items()}
freq_rels = [idx2pred[i] for i in freq_rels_num]

l = pickle.load(open("../em_E.pk", "rb"))
pass
