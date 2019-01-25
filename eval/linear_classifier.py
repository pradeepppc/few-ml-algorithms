import sys
from PIL import Image
import numpy as np

size = 32, 32
N = 32
eta = 0.5
num_iterations = 300
train_file = sys.argv[1]
test_file = sys.argv[2]

with open(train_file, "r") as f:
    lines = f.readlines()
    f.close()

label_set = set()
class_dict = {}
class_dict_index = {}
data_list = []
num_samples = 0
weighted_dict = {}
for idx, l in enumerate(lines):
    img_path, label = l.strip().split()
    if label in label_set:
        class_dict[label] += 1
        class_dict_index[label].append(idx)
    else:
        class_dict[label] = 1
        class_dict_index[label] = [idx]
        label_set.add(label)
        weighted_dict[label] = [0] * N
    num_samples += 1
    im = Image.open(img_path).convert('L')
    im.thumbnail(size, Image.ANTIALIAS)
    pixels = list(im.getdata())
    data_list.append(pixels)
    im.close()

data_array = np.array(data_list)
# mean_array = np.mean(data_array, axis=0)
covar_matrix = np.cov(data_array.T)
eig_value, eig_vect = np.linalg.eigh(covar_matrix)
sorted_indexes = np.argsort(eig_value)
sorted_list = list(sorted_indexes)
sorted_list.reverse()
req_list = sorted_list[0:N]
req_vec = np.array(eig_vect[:, req_list])
reduced_matrix = np.matmul(req_vec.T, data_array.T)
# print(reduced_matrix.shape)
# print(class_dict_index)
# print(class_dict)
# exit()

def run_perceptron(eta, num_iterations, label):
    w = np.array(weighted_dict[label])
    new_w = [1] * N
    sum = [0] * N
    for iter in range(num_iterations):
        for i in range(num_samples):
            output = np.dot(w, reduced_matrix[:, i])
            target = -1
            if i in class_dict_index[label]:
                target = 1
            if output >= 0:
                adder = np.matrix(eta * (target - 1) * reduced_matrix[:, i])
                sum = np.add(sum, adder)
            else:
                adder = np.matrix(eta * (target + 1) * reduced_matrix[:, i])
                sum = np.add(sum, adder)
        new_w = np.add(w, sum)
        w = new_w
    weighted_dict[label] = new_w

for label in label_set:
    run_perceptron(eta, num_iterations, label)
    # print(weighted_dict)


# train a classifier

with open(test_file, "r") as f:
    test_paths = f.readlines()
    f.close()

for p in test_paths:
    # predict the label for image p
    lis = p.strip().split()
    im = Image.open(lis[0]).convert('L')
    im.thumbnail(size, Image.ANTIALIAS)
    pixels = list(im.getdata())
    pixel_array = np.array(pixels)
    comp_image = np.matmul(req_vec.T, pixel_array)
    ans = ''
    maxproduct = -100000000000000
    for label in label_set:
        dpt_prod = np.dot(np.array(weighted_dict[label]), comp_image)
        if dpt_prod > maxproduct:
            maxproduct = dpt_prod
            ans = label
    im.close()
    print(ans)
