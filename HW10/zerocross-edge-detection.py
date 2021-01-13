import cv2
import math
import numpy as np

h = w = 512

def pad(img, pad_size):
	H, W = img.shape
	pad_img = np.zeros((H+pad_size, W+pad_size), np.int32)
	
	for i in range(H):
		pad_img[i+1][0] = img[i][0]
		pad_img[i+1][W+1] = img[i][W-1]

		for j in range(W):
			pad_img[0][j+1] = img[0][j]
			pad_img[H+1][j+1] = img[H-1][j]
			pad_img[i+1][j+1] = img[i][j]

	pad_img[0][0] = img[0][0]
	pad_img[H+1][W+1] = img[H-1][W-1]
	pad_img[0][W+1] = img[0][W-1]
	pad_img[H+1][0] = img[H-1][0]		

	return pad_img

def zeroCross(L):
	edge_img = np.zeros((h, w), np.uint8)
	for i in range(h):
		for j in range(w):
			cross = False
			for m in range(3):
				for n in range(3):
					if not cross and L[i+1][j+1] == 1 and L[i+m][j+n] == -1:
						edge_img[i][j] = 0
						cross = True
					elif not cross:
						edge_img[i][j] = 255
	return edge_img


def Laplacian1(img, thres):
	L = np.zeros((h, w), np.int32)
	mask = np.array([[0, 1, 0],\
					[1, -4, 1],\
					[0, 1, 0]])
	for i in range(h):
		for j in range(w):
			tmp = 0
			for m in range(3):
				for n in range(3):
					tmp += mask[m][n] * img[i+m][j+n]
			if tmp >= thres:
				L[i][j] = 1
			elif tmp <= -thres:
				L[i][j] = -1
			else:
				L[i][j] = 0

	return zeroCross(pad(L, 2))

def Laplacian2(img, thres):
	L = np.zeros((h, w), np.int32)
	mask = np.array([[1, 1, 1],\
					[1, -8, 1],\
					[1, 1, 1]])
	for i in range(h):
		for j in range(w):
			tmp = 0
			for m in range(3):
				for n in range(3):
					tmp += mask[m][n] * img[i+m][j+n]
			if tmp >= thres * 3:
				L[i][j] = 1
			elif tmp <= -thres * 3:
				L[i][j] = -1
			else:
				L[i][j] = 0

	return zeroCross(pad(L, 2))

def minVarLaplac(img, thres):
	L = np.zeros((h, w), np.int32)
	mask = np.array([[2, -1, 2],\
					[-1, -4, -1],\
					[2, -1, 2]])
	for i in range(h):
		for j in range(w):
			tmp = 0
			for m in range(3):
				for n in range(3):
					tmp += mask[m][n] * img[i+m][j+n]
			if tmp >= thres * 3:
				L[i][j] = 1
			elif tmp <= -thres * 3:
				L[i][j] = -1
			else:
				L[i][j] = 0

	return zeroCross(pad(L, 2))

def LoG(img, thres):
	L = np.zeros((h, w), np.int32)
	mask = np.array([[0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],\
					[0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],\
					[0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],\
					[-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],\
					[-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],\
					[-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],\
					[-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],\
					[-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],\
					[0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],\
					[0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],\
					[0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0]])
	for i in range(h):
		for j in range(w):
			tmp = 0
			for m in range(11):
				for n in range(11):
					tmp += mask[m][n] * img[i+m][j+n]
			if tmp >= thres:
				L[i][j] = 1
			elif tmp <= -thres:
				L[i][j] = -1
			else:
				L[i][j] = 0

	return zeroCross(pad(L, 2))

def DoG(img, thres):
	L = np.zeros((h, w), np.int32)
	mask = np.array([[-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],\
					[-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],\
					[-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],\
					[-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],\
					[-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],\
					[-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8],\
					[-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],\
					[-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],\
					[-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],\
					[-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],\
					[-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1]])
	for i in range(h):
		for j in range(w):
			tmp = 0
			for m in range(11):
				for n in range(11):
					tmp += mask[m][n] * img[i+m][j+n]
			if tmp >= thres:
				L[i][j] = 1
			elif tmp <= -thres:
				L[i][j] = -1
			else:
				L[i][j] = 0

	return zeroCross(pad(L, 2))

img_orig = cv2.imread('imgs/lena.bmp', cv2.IMREAD_GRAYSCALE)
img_pad = pad(img_orig, 2)
img_pad5 = pad( pad( pad( pad(img_pad, 2), 2), 2), 2)

cv2.imwrite('imgs/a.jpg', Laplacian1(img_pad, 15))
cv2.imwrite('imgs/b.jpg', Laplacian2(img_pad, 15))
cv2.imwrite('imgs/c.jpg', minVarLaplac(img_pad, 20))
cv2.imwrite('imgs/d.jpg', LoG(img_pad5, 3000))
cv2.imwrite('imgs/e.jpg', DoG(img_pad5, 1))
