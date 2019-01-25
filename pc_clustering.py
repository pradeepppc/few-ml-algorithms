import numpy as np
from sklearn.decomposition import PCA
import os
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

size = 32, 32
N = 3
data_list = []
imageslist = os.listdir('dataset')
for idx, img in enumerate(imageslist):
    image_name = 'dataset/' + img
    im = Image.open(image_name).convert('L')
    im.thumbnail(size, Image.ANTIALIAS)
    pixels = list(im.getdata())
    data_list.append(pixels)

data_array = np.array(data_list)
pca = PCA(n_components=N)
# trans = pd.DataFrame(pca.fit(data_array))
pca.fit(data_array)
X_pca = pca.transform(data_array)
# X_new = pca.inverse_transform(X_pca)
#
# print(X_new[0][:])
fig = plt.figure()
ax = Axes3D(fig)
print(X_pca.shape)
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], cmap='Greens')
plt.show()
