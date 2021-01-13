import cv2
import numpy as np

img = cv2.imread('imgs/lena.bmp')

img_upside_down    = np.ndarray(shape=(img.shape))
img_rightside_left = np.ndarray(shape=(img.shape))
img_diag_mirror    = np.ndarray(shape=(img.shape))

for i in range(511):
	for j in range(511):
		for k in range(3):
			img_upside_down[i][j][k]    = img[511-i][j][k]
			img_rightside_left[i][j][k] = img[i][511-j][k]
			img_diag_mirror[i][j][k]    = img[j][i][k]

cv2.imwrite('imgs/1a.jpg', img_upside_down)
cv2.imwrite('imgs/1b.jpg', img_rightside_left)
cv2.imwrite('imgs/1c.jpg', img_diag_mirror)