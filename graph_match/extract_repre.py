import pickle
import json

import numpy as np
from tqdm import tqdm
import torchvision.models as models
import cv2
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable


def make_model():
    model = models.vgg16(pretrained=True).features[:28]  # 其实就是定位到第28层，对照着上面的key看就可以理解
    model = model.eval()  # 一定要有这行，不然运算速度会变慢（要求梯度）而且会影响结果
    model.cuda()  # 将模型从CPU发送到GPU,如果没有GPU则删除该行
    return model


model = make_model()
num_features = 512
TARGET_IMG_SIZE = 224
img_to_tensor = transforms.ToTensor()


# 特征提取
def extract_feature(model, img):
    model.eval()

    img = cv2.resize(img, (TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    tensor = img_to_tensor(img)
    tensor = tensor.cuda()

    result = model(Variable(tensor))
    pool = torch.nn.MaxPool2d(kernel_size=14, stride=14)
    result = pool(result)
    result = torch.flatten(result)
    result_npy = result.data.cpu().numpy()

    return result_npy


path = "../raw_em_E.pk"

vocab = json.load(open("../VG-SGG-dicts-with-attri.json", "r"))
idx2lb = {int(k): v for k, v in vocab["idx_to_label"].items()}
lb2idx = {k: int(v) for k, v in vocab["label_to_idx"].items()}
idx2pred = {int(k): v for k, v in vocab["idx_to_predicate"].items()}
pred2idx = {k: int(v) for k, v in vocab["predicate_to_idx"].items()}
# labelcnt = {k: int(v) for k, v in vocab["object_count"].items()}
len_lb = len(idx2lb)

l = pickle.load(open(path, "rb"))
lb_count = [0 for _ in range(len_lb)]
lb_feature = np.zeros([len_lb, num_features])

for graph in tqdm(l):
    image = cv2.imread(graph['img_path'])
    boxes = graph['boxes']
    labels = graph['labels']
    for box, label in zip(boxes, labels):
        cropped_image = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        feature = extract_feature(model, cropped_image)
        lb_feature[label - 1] += feature
        lb_count[label - 1] += 1

for i in range(len_lb):
    lb_feature[i] /= lb_count[i]

# for label in labelcnt:
#     print(label)
#     print(labelcnt[label])
#     print(lb_count[lb2idx[label] - 1])
# pass
# print(lb_feature)
np.save('lb_feature', lb_feature)
