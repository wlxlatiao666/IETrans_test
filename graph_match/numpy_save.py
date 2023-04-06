import numpy as np
import json
from scipy import stats

vocab = json.load(open("../VG-SGG-dicts-with-attri.json", "r"))
idx2lb = {int(k): v for k, v in vocab["idx_to_label"].items()}
labelfeature = np.load("lb_feature.npy")
man_feature = labelfeature[78]
for i in range(150):
    other_feature = labelfeature[i]
    s = stats.pearsonr(man_feature, other_feature)
    print(idx2lb[i + 1])
    print(s)
