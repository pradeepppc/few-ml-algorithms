import numpy as np
import sys
from PIL import Image
import math

class_block_num = {}
class_block = {}
class_list = []
data_list = []
num_data = 0
size = 32, 32
arg_list = list(sys.argv)
f = open(arg_list[1], 'r')
f1 = open(arg_list[2], 'r')
lines = f.readlines()
lines1 = f1.readlines()
for idx, line in enumerate(lines):
    lis = line.strip().split()
    if lis[1] in class_list:
        class_block_num[lis[1]] += 1
        class_block[lis[1]].append(idx)
    else:
        class_block_num[lis[1]] = 1
        class_block[lis[1]] = [idx]
        class_list.append(lis[1])
    num_data += 1
    im = Image.open(lis[0]).convert('L')
    im.thumbnail(size, Image.ANTIALIAS)
    pixels = list(im.getdata())
    data_list.append(pixels)
    im.close()
    # print(im.format, im.size, im.mode)

data_array = np.array(data_list)
mean_array = np.mean(data_array, axis=0)
data_array = data_array
covar_matrix = np.cov(data_array.T)
# print(covar_matrix.shape)
# print(covar_matrix[200, 150])
# print(covar_matrix[150, 200])

eig_value, eig_vect = np.linalg.eigh(covar_matrix)
sorted_indexes = np.argsort(eig_value)
sorted_list = list(sorted_indexes)
sorted_list.reverse()
req_list = sorted_list[0:32]
req_vec = np.array(eig_vect[:, req_list])
# print(req_vec.shape)
# print(data_array.shape)
data_comp = np.matmul(req_vec.T, data_array.T)
data_comp = data_comp.T
# print(data_comp.shape)
# exit()
# print(data_array.shape)
class_mean_list = {}
class_variance_list = {}
# print(class_block_num)
# print(class_block)
# print(class_list)
# print(data_comp.shape)

for clas in class_list:
    class_data_list = []
    for n in class_block[clas]:
        class_data_list.append(data_comp[n, :])
    class_data_array = np.array(class_data_list)
    # print(class_data_array.shape)
    class_mean_list[clas] = np.mean(class_data_array, axis=0)
    class_variance_list[clas] = np.var(class_data_array, axis=0)


for index, line in enumerate(lines1):
    lis = line.split()
    im = Image.open(lis[0]).convert('L')
    im.thumbnail(size, Image.ANTIALIAS)
    pixels = list(im.getdata())
    pixel_array = np.array(pixels)
    comp_image = np.matmul(req_vec.T, pixel_array)
    req_pixels = list(comp_image)
    max_val = -10000000
    final_class = ''
    for clas in class_list:
        prod = 1
        for j in range(32):
            value = (1/math.sqrt(2 * math.pi * class_variance_list[clas][j])) * math.exp(-(((req_pixels[j]
                        - class_mean_list[clas][j])**2)/(2 * class_variance_list[clas][j])))
            prod = prod * value
        final_prob = prod * class_block_num[clas]
        # print(final_prob)
        if final_prob > max_val:
            max_val = final_prob
            final_class = clas
    print(final_class)
    im.close()

f.close()
f1.close()

