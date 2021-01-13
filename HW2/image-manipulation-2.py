import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('imgs/lena.bmp', cv2.IMREAD_GRAYSCALE)

img_thresh = np.zeros(shape=(img.shape))
label = np.zeros(shape=(img.shape))
freq = np.zeros(256)
h, w = img.shape

new_label = 1
# 1a, 1b, 1c: binarize & counting & labeling
for i in range(h):
	for j in range(w):
		freq[img[i][j]] += 1
		if img[i][j] < 128:
			img_thresh[i][j] = 0
			label[i][j] = 0
		else:
			img_thresh[i][j] = 255
			label[i][j] = new_label
			new_label += 1
cv2.imwrite('imgs/1a.jpg', img_thresh)

# 1b: histogram
plt.bar(range(0, 256), freq)
plt.savefig('imgs/1b.jpg')

# 1c: Iterative Algorithm, 4-connected
change = True
cnt = 1
while change:
	print(cnt)
	# top-down
	change = False
	for i in range(h):
		for j in range(w):
			if label[i][j] != 0:
				if i != 0 and (0 < label[i-1][j] < label[i][j]):
					label[i][j] = label[i-1][j]
					change = True
				if j != 0 and (0 < label[i][j-1] < label[i][j]):
					label[i][j] = label[i][j-1]
					change = True
	# bottom-up
	for i in range(h-1, 0, -1):
		for j in range(w-1, 0, -1):
			if label[i][j] != 0:
				if i != h-1 and (0 < label[i+1][j] < label[i][j]):
					label[i][j] = label[i+1][j]
					change = True
				if j != w-1 and (0 < label[i][j+1] < label[i][j]):
					label[i][j] = label[i][j+1]
					change = True
	cnt += 1
# count label number
label_cnt = np.zeros(new_label)
for i in range(h):
	for j in range(w):
		label_cnt[int(label[i][j])] += 1

# bounding box & centroid
for k in range(1, new_label):
	if label_cnt[k] >= 500:
		t, b, l, r = h, -1, w, -1
		centroid = [0, 0]
		for i in range(h):
			for j in range(w):
				if label[i][j] == k:
					centroid[0] += i
					centroid[1] += j
					t = min(t, i)
					b = max(b, i)
					l = min(l, j)
					r = max(r, j)
		centroid[0] /= label_cnt[k]
		centroid[1] /= label_cnt[k]
		cv2.rectangle(img_thresh, (l, t), (r, b), 128, 3)
		cv2.circle(img_thresh, (int(centroid[1]), int(centroid[0])), 6, 128, -1)

cv2.imwrite("imgs/1c.jpg", img_thresh)