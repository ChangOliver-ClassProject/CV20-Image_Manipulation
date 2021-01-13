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

def Robert(img, thres):
	edge_img = np.zeros((h, w), np.uint8)
	for i in range(h):
		for j in range(w):
			r1 = img[i+2][j+2] - img[i+1][j+1]
			r2 = img[i+2][j+1] - img[i+1][j+2]
			edge_img[i][j] = 0 if (math.sqrt( math.pow(r1, 2) + math.pow(r2, 2) ) >= thres) else 255

	return edge_img

def Prewitt(img, thres):
	edge_img = np.zeros((h, w), np.uint8)
	for i in range(h):
		for j in range(w):
			p1 = (img[i+2][j] + img[i+2][j+1] + img[i+2][j+2]) - (img[i][j] + img[i][j+1] + img[i][j+2])
			p2 = (img[i][j+2] + img[i+1][j+2] + img[i+2][j+2]) - (img[i][j] + img[i+1][j] + img[i+2][j])
			edge_img[i][j] = 0 if (math.sqrt( math.pow(p1, 2) + math.pow(p2, 2) ) >= thres) else 255			

	return edge_img

def Sobel(img, thres):
	edge_img = np.zeros((h, w), np.uint8)
	for i in range(h):
		for j in range(w):
			s1 = (img[i+2][j] + 2*img[i+2][j+1] + img[i+2][j+2]) - (img[i][j] + 2*img[i][j+1] + img[i][j+2])
			s2 = (img[i][j+2] + 2*img[i+1][j+2] + img[i+2][j+2]) - (img[i][j] + 2*img[i+1][j] + img[i+2][j])
			edge_img[i][j] = 0 if (math.sqrt( math.pow(s1, 2) + math.pow(s2, 2) ) >= thres) else 255			

	return edge_img

def Frei_n_Chen(img, thres):
	edge_img = np.zeros((h, w), np.uint8)
	for i in range(h):
		for j in range(w):
			f1 = (img[i+2][j] + math.sqrt(2)*img[i+2][j+1] + img[i+2][j+2]) - (img[i][j] + math.sqrt(2)*img[i][j+1] + img[i][j+2])
			f2 = (img[i][j+2] + math.sqrt(2)*img[i+1][j+2] + img[i+2][j+2]) - (img[i][j] + math.sqrt(2)*img[i+1][j] + img[i+2][j])
			edge_img[i][j] = 0 if (math.sqrt( math.pow(f1, 2) + math.pow(f2, 2) ) >= thres) else 255			

	return edge_img

def Kirsch_Compass(img, thres):
	edge_img = np.zeros((h, w), np.uint8)
	k = [0] * 8
	for i in range(h):
		for j in range(w):
			k[0] = 5*(img[i][j+2] + img[i+1][j+2] + img[i+2][j+2]) - 3*(img[i][j] + img[i][j+1] + img[i+1][j] + img[i+2][j] + img[i+2][j+1])
			k[1] = 5*(img[i][j+2] + img[i+1][j+2] + img[i][j+1]) - 3*(img[i][j] + img[i+1][j] + img[i+2][j] + img[i+2][j+1] + img[i+2][j+2])
			k[2] = 5*(img[i][j] + img[i][j+1] + img[i][j+2]) - 3*(img[i+1][j] + img[i+2][j] + img[i+2][j+1] + img[i+1][j+2] + img[i+2][j+2])
			k[3] = 5*(img[i][j] + img[i+1][j] + img[i][j+1]) - 3*(img[i][j+2] + img[i+1][j+2] + img[i+2][j] + img[i+2][j+1] + img[i+2][j+2])
			k[4] = 5*(img[i][j] + img[i+1][j] + img[i+2][j]) - 3*(img[i][j+1] + img[i][j+2] + img[i+1][j+2] + img[i+2][j+1] + img[i+2][j+2])
			k[5] = 5*(img[i+1][j] + img[i+2][j] + img[i+2][j+1]) - 3*(img[i][j] + img[i][j+1] + img[i][j+2] + img[i+1][j+2] + img[i+2][j+2])
			k[6] = 5*(img[i+2][j] + img[i+2][j+1] + img[i+2][j+2]) - 3*(img[i][j] + img[i][j+1] + img[i][j+2] + img[i+1][j] + img[i+1][j+2])
			k[7] = 5*(img[i+2][j+1] + img[i+2][j+2] + img[i+1][j+2]) - 3*(img[i][j] + img[i][j+1] + img[i][j+2] + img[i+1][j] + img[i+2][j])
			edge_img[i][j] = 0 if (max(k) >= thres) else 255

	return edge_img

def Robinson_Compass(img, thres):
	edge_img = np.zeros((h, w), np.uint8)
	r = [0] * 8
	for i in range(h):
		for j in range(w):
			r[0] = (img[i][j+2] + 2*img[i+1][j+2] + img[i+2][j+2]) - (img[i][j] + 2*img[i+1][j] + img[i+2][j])
			r[1] = (img[i][j+1] + 2*img[i][j+2] + img[i+1][j+2]) - (img[i+1][j] + 2*img[i+2][j] + img[i+2][j+1])
			r[2] = (img[i][j] + 2*img[i][j+1] + img[i][j+2]) - (img[i+2][j] + 2*img[i+2][j+1] + img[i+2][j+2])
			r[3] = (img[i+1][j] + 2*img[i][j] + img[i][j+1]) - (img[i+2][j+1] + 2*img[i+2][j+2] + img[i+1][j+2])
			r[4] = (img[i][j] + 2*img[i+1][j] + img[i+2][j]) - (img[i][j+2] + 2*img[i+1][j+2] + img[i+2][j+2])
			r[5] = (img[i+1][j] + 2*img[i+2][j] + img[i+2][j+1]) - (img[i][j+1] + 2*img[i][j+2] + img[i+1][j+2])
			r[6] = (img[i+2][j] + 2*img[i+2][j+1] + img[i+2][j+2]) - (img[i][j] + 2*img[i][j+1] + img[i][j+2])
			r[7] = (img[i+1][j+2] + 2*img[i+2][j+2] + img[i+2][j+1]) - (img[i+1][j] + 2*img[i][j] + img[i][j+1])
			edge_img[i][j] = 0 if (max(r) >= thres) else 255

	return edge_img

def Nevatia_Babu_5x5(img, thres):
	edge_img = np.zeros((h, w), np.uint8)
	n0 = np.array([[100, 100, 100, 100, 100],\
					[100, 100, 100, 100, 100],\
					[0, 0, 0, 0, 0],\
					[-100, -100, -100, -100, -100],\
					[-100, -100, -100, -100, -100]])
	n1 = np.array([[100, 100, 100, 100, 100],\
					[100, 100, 100, 78, -32],\
					[100, 92, 0, -92, -100],\
					[32, -78, -100, -100, -100],\
					[-100, -100, -100, -100, -100]])
	n2 = np.array([[100, 100, 100, 32, -100],\
					[100, 100, 92, -78, -100],\
					[100, 100, 0, -100, -100],\
					[100, 78, -92, -100, -100],\
					[100, -32, -100, -100, -100]])
	n3 = np.array([[-100, -100, 0, 100, 100],\
					[-100, -100, 0, 100, 100],\
					[-100, -100, 0, 100, 100],\
					[-100, -100, 0, 100, 100],\
					[-100, -100, 0, 100, 100]])
	n4 = np.array([[-100, 32, 100, 100, 100],\
					[-100, -78, 92, 100, 100],\
					[-100, -100, 0, 100, 100],\
					[-100, -100, -92, 78, 100],\
					[-100, -100, -100, -32, 100]])
	n5 = np.array([[100, 100, 100, 100, 100],\
					[-32, 78, 100, 100, 100],\
					[-100, -92, 0, 92, 100],\
					[-100, -100, -100, -78, 32],\
					[-100, -100, -100, -100, -100]])
	
	for i in range(h):
		for j in range(w):
			N = [0] * 6
			for m in range(5):
				for n in range(5):
					N[0] += n0[m][n] * img[i+m][j+n] 
					N[1] += n1[m][n] * img[i+m][j+n] 
					N[2] += n2[m][n] * img[i+m][j+n] 
					N[3] += n3[m][n] * img[i+m][j+n] 
					N[4] += n4[m][n] * img[i+m][j+n] 
					N[5] += n5[m][n] * img[i+m][j+n] 
			edge_img[i][j] = 0 if (max(N) >= thres) else 255	
				
	return edge_img

img_orig = cv2.imread('imgs/lena.bmp', cv2.IMREAD_GRAYSCALE)
img_pad = pad(img_orig, 2)
img_pad2 = pad(img_pad, 2)

cv2.imwrite('imgs/a.jpg', Robert(img_pad, 12))
cv2.imwrite('imgs/b.jpg', Prewitt(img_pad, 24))
cv2.imwrite('imgs/c.jpg', Sobel(img_pad, 38))
cv2.imwrite('imgs/d.jpg', Frei_n_Chen(img_pad, 30))
cv2.imwrite('imgs/e.jpg', Kirsch_Compass(img_pad, 135))
cv2.imwrite('imgs/f.jpg', Robinson_Compass(img_pad, 43))
cv2.imwrite('imgs/g.jpg', Nevatia_Babu_5x5(img_pad2, 12500))
