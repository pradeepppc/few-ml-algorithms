import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt


size = 32, 32
# N = 2
data_list = []
imageslist = os.listdir('dataset')
for idx, img in enumerate(imageslist):
    image_name = 'dataset/' + img
    im = Image.open(image_name).convert('L')
    im.thumbnail(size, Image.ANTIALIAS)
    pixels = list(im.getdata())
    data_list.append(pixels)
x_list = []
y_list = []
data_arrays = np.array(data_list)
mean_array = np.mean(data_arrays, axis=0)
data_array = data_arrays
covar_matrix = np.cov(data_array.T)
eig_value, eig_vect = np.linalg.eigh(covar_matrix)
sorted_indexes = np.argsort(eig_value)
sorted_list = list(sorted_indexes)
sorted_list.reverse()
for N in range(519, len(imageslist), 30):
    first_img = data_array[0, :]

    req_list = sorted_list[0:N]
    req_vec = np.array(eig_vect[:, req_list])

    # print(req_vec.shape)
    reduced_images_data = np.matmul(req_vec.T, (data_array).T)
    reduced_images_data = reduced_images_data.T
    # print(reduced_images_data.shape)
    reconstructed_image_data = np.matmul(reduced_images_data, req_vec.T)
    # print(reconstructed_image_data.shape)
    # print(req_vec.shape)
    # print(len(req_list))
    first_img_data = reconstructed_image_data[0, :]
    first_img_data = np.float64(first_img_data)
    print(first_img_data)
    fig = plt.figure(figsize=(1, 2))
    fig.add_subplot(1, 2, 1)
    plt.imshow(np.reshape(first_img, size), cmap=plt.cm.bone, interpolation='nearest')
    plt.title('Original Image')
    fig.add_subplot(1, 2, 2)
    plt.imshow(np.reshape(first_img_data, size), cmap=plt.cm.bone, interpolation='nearest')
    plt.title('Reconstructed image')
    plt.show()
    exit()

    # data_array = data_array + mean_array

    error = 0
    for i in range(data_array.shape[0]):
        for j in range(data_array.shape[1]):
            error = error + (data_arrays[i][j] - reconstructed_image_data[i][j])**2
    m_s_e = error/(data_array.shape[0] * data_array.shape[1])
    x_list.append(N)
    y_list.append(m_s_e)

plt.plot(x_list, y_list)
plt.title('Graph of mean square error vs principal components taken for all samples')
plt.xlabel('Number of principal components taken')
plt.ylabel('mean square error')
plt.show()
