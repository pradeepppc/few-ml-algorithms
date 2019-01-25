line1 = open('answer').readlines()
line2 = open('test_ans.txt').readlines()
W = 0
R = 0
for l1, l2 in zip(line1, line2):
	if l1.strip() == l2.strip():
		R += 1
	else:
		W += 1
print(R, W, R+W)