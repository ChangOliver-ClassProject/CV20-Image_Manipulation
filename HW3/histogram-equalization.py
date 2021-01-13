import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('imgs/lena.bmp', cv2.IMREAD_GRAYSCALE)

img_divide = np.ndarray(img.shape)
freq_original = np.zeros(256)
freq_divide = np.zeros(256)
h, w = img.shape

# 1a, 1b: original & modified frequency count
for i in range(h):
	for j in range(w):
		img_divide[i][j] = img[i][j] // 3
		freq_original[img[i][j]] += 1
		freq_divide[img[i][j] // 3] += 1

# 1a original histogram
plt.figure(0)
plt.bar(range(0, 256), freq_original)
plt.savefig('imgs/1a_hist.jpg')

# 1b modified histogram
plt.figure(1)
plt.bar(range(0, 256), freq_divide)
plt.savefig('imgs/1b_hist.jpg')
cv2.imwrite('imgs/1b_img.jpg', img_divide)

# 1c histogram equalization
img_equalize = np.ndarray(img.shape)
freq_equalize = np.zeros(256)
s = np.zeros(256, dtype=np.int)
total = 0
for i in range(256):
	total += freq_divide[i]
	s[i] = (total / (h*w)) * 255

for i in range(h):
	for j in range(w):
		img_equalize[i][j] = s[img[i][j] // 3]
		freq_equalize[s[img[i][j] // 3]] += 1
		
np.savetxt('histogram.csv', freq_equalize, fmt='%d', delimiter=',')
plt.figure(2)
plt.bar(range(0, 256), freq_equalize)
plt.savefig('imgs/1c_hist.jpg')
cv2.imwrite('imgs/1c_img.jpg', img_equalize)
