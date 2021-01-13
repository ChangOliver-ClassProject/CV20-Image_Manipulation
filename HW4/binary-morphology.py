import cv2
import numpy as np

img = cv2.imread('imgs/lena.bmp', cv2.IMREAD_GRAYSCALE)

img_thresh = np.zeros(shape=(img.shape), dtype=np.uint8)
kernel = np.ones((5,5), dtype=np.uint8)
kernel[0][0] = kernel[0][4] = kernel[4][0] = kernel[4][4] = 0
h, w = img.shape

# binarize
for i in range(h):
	for j in range(w):
		img_thresh[i][j] = 0 if img[i][j] < 128 else 255


def dilate(img, kernel):
	img_dilate = np.zeros(shape=(img.shape) , dtype=np.uint8)
	for i in range(h):
		for j in range(w):
			if img[i][j]:
				for m in range(-2, 3):
					for n in range(-2, 3):
						if 0 <= i+m < h and 0 <= j+n < w and kernel[m+2][n+2]:
							img_dilate[i+m][j+n] = 255	
	return img_dilate

def erode(img, kernel):
	img_erode = np.full(shape=(img.shape), fill_value=255, dtype=np.uint8)
	for i in range(h):
		for j in range(w):
			if img[i][j]:
				for m in range(-2, 3):
					for n in range(-2, 3):
						if 0 > i+m or i+m >= h or 0 > j+n or j+n >= w:
							img_erode[i][j] = 0
						elif 0 <= i+m < h and 0 <= j+n < w and kernel[m+2][n+2] and not img[i+m][j+n]:
							img_erode[i][j] = 0
			else:
				img_erode[i][j] = 0
	return img_erode

# (a) dilation
img_dilate = dilate(img_thresh, kernel)
cv2.imwrite('imgs/a.jpg', img_dilate)

# (b) erosion
img_erode = erode(img_thresh, kernel)
cv2.imwrite('imgs/b.jpg', img_erode)

# (c) opening
img_open = dilate(img_erode, kernel)
cv2.imwrite('imgs/c.jpg', img_open)

# (d) closing
img_close = erode(img_dilate, kernel)
cv2.imwrite('imgs/d.jpg', img_close)

# (e) hit-and-miss
img_complement = 255 - img_thresh
img_hit_miss = np.zeros(shape=(img.shape), dtype=np.uint8)

for i in range(h):
	for j in range(w):
		if i+1 < h and j-1 >= 0 and i - 1 >= 0 and j+1 < w and \
		img_thresh[i][j] == img_thresh[i+1][j] == img_thresh[i][j-1] == \
		img_complement[i-1][j] == img_complement[i-1][j+1] == img_complement[i][j+1] == 255:
			img_hit_miss[i][j] = 255

cv2.imwrite('imgs/e.jpg', img_hit_miss)