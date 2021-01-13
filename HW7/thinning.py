import cv2
import numpy as np

def h_yokoi(b, c, d, e):
	if b == c and (d != b or e != b):
		return 'q'
	elif b == c == d == e:
		return 'r'
	elif b != c:
		return 's'

def f_yokoi(a1, a2, a3, a4):
	if a1 == a2 == a3 == a4 == 'r':
		return 5
	else:
		return [a1, a2, a3, a4].count('q')

def h_cs(b, c, d, e):
	if b == c and (d != b or e != b):
		return 1
	else:
		return 0

def f_cs(a1, a2, a3, a4, x0):
	if [a1, a2, a3, a4].count(1) == 1:
		return 0
	else:
		return x0

h, w = 64, 64
img = cv2.imread('imgs/lena.bmp', cv2.IMREAD_GRAYSCALE)
img_downsample = np.zeros(shape=(h, w), dtype=np.uint8)

# binarize & downsample
for i in range(h):
	for j in range(w):
		img_downsample[i][j] = 255 if (img[i*8][j*8] >= 128) else 0

######################
# x[7] # x[2] # x[6] #
# x[3] # x[0] # x[1] #
# x[8] # x[4] # x[5] #
######################
change = True
while change:
	change = False
	x = np.zeros(9, dtype=np.uint8)
	yokoi = np.zeros(shape=(h, w), dtype=np.uint8)

	for i in range(h):
		for j in range(w):
			if img_downsample[i][j] == 0:
				continue

			x[0] = img_downsample[i][j]
			x[1] = img_downsample[i][j+1] if (j + 1 < w) else 0
			x[2] = img_downsample[i-1][j] if (i - 1 >= 0) else 0
			x[3] = img_downsample[i][j-1] if (j -1 >= 0) else 0
			x[4] = img_downsample[i+1][j] if (i +1 < h) else 0
			x[5] = img_downsample[i+1][j+1] if (i + 1 < h and j + 1 < w) else 0
			x[6] = img_downsample[i-1][j+1] if (i - 1 >= 0 and j + 1 < w) else 0
			x[7] = img_downsample[i-1][j-1] if (i - 1 >= 0 and j - 1 >= 0) else 0
			x[8] = img_downsample[i+1][j-1] if (i + 1 < h and j - 1 >= 0) else 0

			yokoi[i][j] = f_yokoi(h_yokoi(x[0], x[1], x[6], x[2]),\
							 h_yokoi(x[0], x[2], x[7], x[3]), \
							 h_yokoi(x[0], x[3], x[8], x[4]), \
							 h_yokoi(x[0], x[4], x[5], x[1]))

	pr = np.full(shape=(h, w), fill_value='g', dtype=str)
	for i in range(h):
		for j in range(w):
			if img_downsample[i][j] == 0:
				continue
			elif yokoi[i][j] != 1:
				pr[i][j] = 'q'
				continue

			x[1] = 1 if (j + 1 < w) and yokoi[i][j+1] == 1 else 0
			x[2] = 1 if (i - 1 >= 0) and yokoi[i-1][j] == 1 else 0
			x[3] = 1 if (j -1 >= 0) and yokoi[i][j-1] == 1 else 0
			x[4] = 1 if (i +1 < h) and yokoi[i+1][j] == 1 else 0

			pr[i][j] = 'p' if x[1] + x[2] + x[3] + x[4] >= 1 else 'q'
	
	for i in range(h):
		for j in range(w):
			if pr[i][j] == 'q' or img_downsample[i][j] == 0:
				continue

			x[0] = img_downsample[i][j]
			x[1] = img_downsample[i][j+1] if (j + 1 < w) else 0
			x[2] = img_downsample[i-1][j] if (i - 1 >= 0) else 0
			x[3] = img_downsample[i][j-1] if (j -1 >= 0) else 0
			x[4] = img_downsample[i+1][j] if (i +1 < h) else 0
			x[5] = img_downsample[i+1][j+1] if (i + 1 < h and j + 1 < w) else 0
			x[6] = img_downsample[i-1][j+1] if (i - 1 >= 0 and j + 1 < w) else 0
			x[7] = img_downsample[i-1][j-1] if (i - 1 >= 0 and j - 1 >= 0) else 0
			x[8] = img_downsample[i+1][j-1] if (i + 1 < h and j - 1 >= 0) else 0

			value = f_cs(h_cs(x[0], x[1], x[6], x[2]),\
							 h_cs(x[0], x[2], x[7], x[3]), \
							 h_cs(x[0], x[3], x[8], x[4]), \
							 h_cs(x[0], x[4], x[5], x[1]), x[0])
	
			if value != x[0]:
				img_downsample[i][j] = value
				change = True
	
cv2.imwrite('imgs/thin.jpg', img_downsample)