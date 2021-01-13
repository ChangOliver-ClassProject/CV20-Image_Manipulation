import cv2
import numpy as np

img = cv2.imread('imgs/lena.bmp', cv2.IMREAD_GRAYSCALE)

kernel = np.ones((5,5), dtype=np.uint8)
kernel[0][0] = kernel[0][4] = kernel[4][0] = kernel[4][4] = 0
h, w = img.shape

def dilate(img, kernel):
	img_dilate = np.zeros(shape=(img.shape) , dtype=np.uint8)
	for i in range(h):
		for j in range(w):
			if img[i][j]:
				max_val = 0
				for m in range(-2, 3):
					for n in range(-2, 3):
						if 0 <= i+m < h and 0 <= j+n < w and kernel[m+2][n+2] and img[i+m][j+n] > max_val:
							max_val = img[i+m][j+n]
				img_dilate[i][j] = max_val	

	return img_dilate

def erode(img, kernel):
	img_erode = np.zeros(shape=(img.shape) , dtype=np.uint8)
	for i in range(h):
		for j in range(w):
			if img[i][j]:
				min_val = 256
				for m in range(-2, 3):
					for n in range(-2, 3):
						if 0 <= i+m < h and 0 <= j+n < w and kernel[m+2][n+2] and img[i+m][j+n] < min_val:
							min_val = img[i+m][j+n]
				img_erode[i][j] = min_val
				
	return img_erode

# (a) dilation
img_dilate = dilate(img, kernel)
cv2.imwrite('imgs/a.jpg', img_dilate)

# (b) erosion
img_erode = erode(img, kernel)
cv2.imwrite('imgs/b.jpg', img_erode)

# (c) opening
img_open = dilate(img_erode, kernel)
cv2.imwrite('imgs/c.jpg', img_open)

# (d) closing
img_close = erode(img_dilate, kernel)
cv2.imwrite('imgs/d.jpg', img_close)