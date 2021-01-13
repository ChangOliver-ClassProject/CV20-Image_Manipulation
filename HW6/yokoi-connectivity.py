import cv2
import numpy as np

def H(b, c, d, e):
	if b == c and (d != b or e != b):
		return 'q'
	elif b == c == d == e:
		return 'r'
	elif b != c:
		return 's'

def f(a1, a2, a3, a4):
	if a1 == a2 == a3 == a4 == 'r':
		return 5
	else:
		return [a1, a2, a3, a4].count('q')

img = cv2.imread('imgs/lena.bmp', cv2.IMREAD_GRAYSCALE)
img_downsample = np.zeros(shape=(64, 64), dtype=np.uint8)
h, w = 64, 64

# binarize & downsample
for i in range(h):
	for j in range(w):
		img_downsample[i][j] = 255 if (img[i*8][j*8] >= 128) else 0


######################
# x[7] # x[2] # x[6] #
# x[3] # x[0] # x[1] #
# x[8] # x[4] # x[5] #
######################
x = np.zeros(9, dtype=np.uint8)
yokoi = np.zeros(shape=(64, 64), dtype=np.uint8)

for i in range(h):
	for j in range(w):
		if img_downsample[i][j] == 0:
			continue;

		x[0] = img_downsample[i][j]
		x[1] = img_downsample[i][j+1] if (j + 1 < w) else 0
		x[2] = img_downsample[i-1][j] if (i - 1 >= 0) else 0
		x[3] = img_downsample[i][j-1] if (j -1 >= 0) else 0
		x[4] = img_downsample[i+1][j] if (i +1 < h) else 0
		x[5] = img_downsample[i+1][j+1] if (i + 1 < h and j + 1 < w) else 0
		x[6] = img_downsample[i-1][j+1] if (i - 1 >= 0 and j + 1 < w) else 0
		x[7] = img_downsample[i-1][j-1] if (i - 1 >= 0 and j - 1 >= 0) else 0
		x[8] = img_downsample[i+1][j-1] if (i + 1 < h and j - 1 >= 0) else 0

		yokoi[i][j] = f(H(x[0], x[1], x[6], x[2]),\
						 H(x[0], x[2], x[7], x[3]), \
						 H(x[0], x[3], x[8], x[4]), \
						 H(x[0], x[4], x[5], x[1]))

for i in range(h):
	for j in range(w):
		if j == w-1:
			print(' ' if yokoi[i][j] == 0 else yokoi[i][j], end='\n')
		else:
			print(' ' if yokoi[i][j] == 0 else yokoi[i][j], end='')
