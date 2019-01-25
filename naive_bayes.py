import numpy as np
import sys
from PIL import Image
import math

class_block_num = {}
class_block = {}
class_list = []
data_list = []
num_data = 0
arg_list = list(sys.argv)
f = open(arg_list[1], 'r')
f1 = open(arg_list[2], 'r')
lines = f.readlines()
lines1 = f1.readlines()
for idx, line in enumerate(lines):
    lis = line.split()
    if lis[1] in class_list:
        class_block_num[lis[1]] += 1
        class_block[lis[1]].append(idx)
    else:
        class_block_num[lis[1]] = 1
        class_block[lis[1]] = [idx]
        class_list.append(lis[1])
    num_data += 1
    im = Image.open(lis[0]).convert('L')
    pixels = list(im.getdata())
    data_list.append(pixels)
    im.close()
    # print(im.format, im.size, im.mode)

data_array = np.array(data_list)
mean_array = np.mean(data_array, axis=0)
data_array = data_array - mean_array
covariance_list = []
for i in range(data_array.shape[1]):
    sum = 0
    for j in range(data_array.shape[0]):
        sum = sum + (data_array[j, i])**2
    covariance_list.append(sum)
sorted_index = np.argsort(covariance_list)
sorted_index = sorted_index.tolist()
sorted_index.reverse()
n_sorted_index = sorted_index[0:32]
# print(n_sorted_index)
# print(data_array.shape[1])
# print(class_list)
# print(class_block_num)
# print(class_block)
class_mean_list = {}
class_variance_list = {}

for clas in class_list:
    class_data_list = []
    for n in class_block[clas]:
        class_data_list.append(data_array[n, n_sorted_index])
    class_data_array = np.array(class_data_list)
    # print(class_data_array)
    class_mean_list[clas] = np.mean(class_data_array, axis=0)
    class_variance_list[clas] = np.var(class_data_array, axis=0)

# print(len(class_mean_list['bob']))
# print(len(class_variance_list['bob']))

for index, line in enumerate(lines1):
    lis = line.split()
    im = Image.open(lis[0]).convert('L')
    pixels = list(im.getdata())
    req_pixels = [pixels[x] for x in n_sorted_index]
    max_val = -1000
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
