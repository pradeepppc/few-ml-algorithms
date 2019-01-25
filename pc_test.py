import numpy as np
from sklearn.decomposition import PCA
import os
from PIL import Image
import matplotlib.pyplot as plt

size = 256, 256
N = 20
data_list = []
imageslist = os.listdir('dataset')
for idx, img in enumerate(imageslist):
    image_name = 'dataset/' + img
    im = Image.open(image_name).convert('L')
    # im.thumbnail(size, Image.ANTIALIAS)
    pixels = list(im.getdata())
    data_list.append(pixels)

data_array = np.array(data_list)
mean_array = np.mean(data_array, axis=0)
first_img = data_array[0, :]
data_array = data_array
covar_data = np.cov(data_array)

eig_value, eig_vect = np.linalg.eig(covar_data)
sorted_indexes = np.argsort(eig_value)
sorted_list = list(sorted_indexes)
sorted_list.reverse()
req_list = sorted_list[0:N]
req_vec = np.array(eig_vect[:, req_list])
req_vec = np.matmul(data_array.T, req_vec)
final_array = np.matmul(req_vec.T, data_array.T)
final_array = final_array.T
reconstructed_image_data = np.matmul(final_array, req_vec.T)
first_img_data = reconstructed_image_data[0, :]
first_img_data = np.float64(first_img_data)

fig = plt.figure(figsize=(1, 2))
fig.add_subplot(1, 2, 1)
plt.imshow(np.reshape(first_img, size), cmap=plt.cm.bone, interpolation='nearest')
fig.add_subplot(1, 2, 2)
plt.imshow(np.reshape(first_img_data, size), cmap=plt.cm.bone, interpolation='nearest')
plt.show()

print(final_array.shape)
print(req_vec.shape)
print(data_array.shape)
print(covar_data.shape)


