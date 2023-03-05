from sklearn.decomposition import PCA
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image

im = cv.imread('test1.jpg')
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
pca1 = PCA(n_components=2)
p = pca1.fit_transform(imgray)
print(pca1.explained_variance_ratio_)
p=p.flatten()
pca2 = PCA(n_components=10)
pp = pca2.fit_transform(p)
# plt.imshow(im, 'gray')
# plt.show()
pass
