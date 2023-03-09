import cv2
from PIL import Image
import pickle
import json
import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable
import torch.cuda
import torchvision.transforms as transforms
import numpy as np

test_graph_num = 4784
num_features = 512
TARGET_IMG_SIZE = 224
img_to_tensor = transforms.ToTensor()


def make_model():
    model = models.vgg16(pretrained=True).features[:28]  # 其实就是定位到第28层，对照着上面的key看就可以理解
    model = model.eval()  # 一定要有这行，不然运算速度会变慢（要求梯度）而且会影响结果
    model.cuda()  # 将模型从CPU发送到GPU,如果没有GPU则删除该行
    return model


# 特征提取
def extract_feature(model, img):
    model.eval()  # 必须要有，不然会影响特征提取结果

    img = cv2.resize(img, (TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    tensor = img_to_tensor(img)  # 将图片转化成tensor
    tensor = tensor.cuda()  # 如果只是在cpu上跑的话要将这行去掉

    result = model(Variable(tensor))
    pool = torch.nn.MaxPool2d(kernel_size=14, stride=14)
    result = pool(result)
    result = torch.flatten(result)
    result_npy = result.data.cpu().numpy()  # 保存的时候一定要记得转成cpu形式的，不然可能会出错

    return result_npy


vocab = json.load(open("../VG-SGG-dicts-with-attri.json", "r"))
idx2lb = {int(k): v for k, v in vocab["idx_to_label"].items()}
path = "../em_E.pk"
l = pickle.load(open(path, "rb"))
print(l[test_graph_num])

boxes = l[test_graph_num]['boxes']
# box=boxes[0]
# num=box[0]
labels = l[test_graph_num]['labels']

img = cv2.imread(l[test_graph_num]['img_path'])

# Prints Dimensions of the image
# print(img.shape)

# Display the image
# cv2.imshow("original", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# model = make_model()
# node1 = []
for i, box in enumerate(boxes):
    if i==6:
        box[0]=0
    cropped_image = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]  # Slicing to crop the image
    # node_feature = extract_feature(model, cropped_image)
    # node1.append(node_feature)
    # Display the cropped image
    if i==5:
        cv2.imshow(idx2lb[labels[i]], cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
# node1 = np.array(node1)
# print(node1)
pass
