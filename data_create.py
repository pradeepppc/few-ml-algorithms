import os, random
files = []
for fi in os.listdir('./dataset'):
	files.append(fi)
fil = open('train.txt', 'w')
F = open('test.txt', 'w')
FA = open('test_ans.txt','w')
train = []
for i in range(400):
	item = random.choice(files)
	train.append(item)
	files.remove(item)
for fi in train:
	fil.write('./dataset/'+fi+' '+fi.split('_')[0]+'\n')
for i in range(len(files)):
	if i < 100:
		F.write('./dataset/'+files[i]+'\n')
		FA.write(files[i].split('_')[0]+'\n')