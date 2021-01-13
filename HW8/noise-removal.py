import cv2
import math
import numpy as np

h, w = 512, 512
kernel = np.ones((5,5), dtype=np.uint8)
kernel[0][0] = kernel[0][4] = kernel[4][0] = kernel[4][4] = 0

def SNR(img, img_noise):
	ms = vs = mn = vn = 0
	for i in range(h):
		for j in range(w):
			ms += img[i][j]
			mn += (img_noise[i][j] - img[i][j])
	ms /= (h*w)
	mn /= (h*w)

	for i in range(h):
		for j in range(w):
			vs += math.pow(img[i][j] - ms, 2)
			vn += math.pow(img_noise[i][j] - img[i][j] - mn, 2)
	vs /= (h*w)
	vn /= (h*w)
	
	return 20 * math.log(math.sqrt(vs) / math.sqrt(vn), 10)

def gaussian_noise(pixel, amp):
	noisePixel = int(pixel + amp * np.random.normal())
	return 255 if noisePixel > 255 else noisePixel

def salt_n_pepper(pixel, thres):
	v = np.random.uniform()
	if v < thres:
		return 0
	elif v > 1 - thres:
		return 255
	else:
		return pixel

def box_filter(img, size):
	img_box = np.zeros(shape=(h, w), dtype=np.uint)
	offset = size // 2
	for i in range(h):
		for j in range(w):
			total = count = 0
			for m in range(-offset, offset+1):
				for n in range(-offset, offset+1):
					if (0 <= i+m < h) and (0 <= j+n < w):
						total += img[i+m][j+n]
						count += 1
			img_box[i][j] = total // count
	
	return img_box

def median_filter(img, size):
	img_median = np.zeros(shape=(h, w), dtype=np.uint)
	offset = size // 2
	for i in range(h):
		for j in range(w):
			values = []
			for m in range(-offset, offset+1):
				for n in range(-offset, offset+1):
					if (0 <= i+m < h) and (0 <= j+n < w):
						values.append(img[i+m][j+n])
			values.sort()
			img_median[i][j] = values[len(values) // 2]
	
	return img_median

def dilate(img, kernel, k_center = None):
    if k_center == None:
        k_center = [x // 2 for x in kernel.shape]
    ret = np.zeros(img.shape)
    for i in range(h):
        for j in range(w):
            # for each pixel in binary image
            brightest = 0
            for ki in range(kernel.shape[0]):
                for kj in range(kernel.shape[1]):
                    if kernel[ki][kj] == 1:
                        dest_X = i + (ki - k_center[0])
                        dest_Y = j + (kj - k_center[1])
                        if 0 <= dest_X < h and 0 <= dest_Y < w:
                            brightest = max(brightest, img[dest_X][dest_Y])
            ret[i][j] = brightest
    return ret

def erode(img, kernel, k_center = None):
    if k_center == None:
        k_center = [x // 2 for x in kernel.shape]
    ret = np.zeros(img.shape)
    for i in range(h):
        for j in range(w):
            darkest = 255
            for ki in range(kernel.shape[0]):
                for kj in range(kernel.shape[1]):
                    if kernel[ki][kj] == 1:
                        dest_X = i + (ki - k_center[0])
                        dest_Y = j + (kj - k_center[1])
                        if 0 <= dest_X < h and 0 <= dest_Y < w:
                            darkest = min(darkest, img[dest_X][dest_Y])
            ret[i][j] = darkest
    return ret	

def open(img, kernel):
	return dilate(erode(img, kernel), kernel)

def close(img, kernel):
	return erode(dilate(img, kernel), kernel)

img = cv2.imread('imgs/lena.bmp', cv2.IMREAD_GRAYSCALE)
img_gaussian_10 = np.zeros(shape=(h, w), dtype=np.uint8)
img_gaussian_30 = np.zeros(shape=(h, w), dtype=np.uint8)
img_snp_01 = np.zeros(shape=(h, w), dtype=np.uint8)
img_snp_005 = np.zeros(shape=(h, w), dtype=np.uint8)

# generate noise
for i in range(h):
	for j in range(w):
		img_gaussian_10[i][j] = gaussian_noise(img[i][j], 10)
		img_gaussian_30[i][j] = gaussian_noise(img[i][j], 30)
		img_snp_01[i][j] = salt_n_pepper(img[i][j], 0.1)
		img_snp_005[i][j] = salt_n_pepper(img[i][j], 0.05)	

#3x3 box_filter
img_gaussian_10_3box = box_filter(img_gaussian_10, 3)
img_gaussian_30_3box = box_filter(img_gaussian_30, 3)
img_snp_01_3box = box_filter(img_snp_01, 3)
img_snp_005_3box = box_filter(img_snp_005, 3)

#5x5 box_filter
img_gaussian_10_5box = box_filter(img_gaussian_10, 5)
img_gaussian_30_5box = box_filter(img_gaussian_30, 5)
img_snp_01_5box = box_filter(img_snp_01, 5)
img_snp_005_5box = box_filter(img_snp_005, 5)

#3x3 median_filter
img_gaussian_10_3med = median_filter(img_gaussian_10, 3)
img_gaussian_30_3med = median_filter(img_gaussian_30, 3)
img_snp_01_3med = median_filter(img_snp_01, 3)
img_snp_005_3med = median_filter(img_snp_005, 3)

#5x5 median_filter
img_gaussian_10_5med = median_filter(img_gaussian_10, 5)
img_gaussian_30_5med = median_filter(img_gaussian_30, 5)
img_snp_01_5med = median_filter(img_snp_01, 5)
img_snp_005_5med = median_filter(img_snp_005, 5)

# openning then closing
img_gaussian_10_openclose = close(open(img_gaussian_10, kernel), kernel)
img_gaussian_30_openclose = close(open(img_gaussian_30, kernel), kernel)
img_snp_01_openclose = close(open(img_snp_01, kernel), kernel)
img_snp_005_openclose = close(open(img_snp_005, kernel), kernel)

# closing then openning
img_gaussian_10_closeopen = open(close(img_gaussian_10, kernel), kernel)
img_gaussian_30_closeopen = open(close(img_gaussian_30, kernel), kernel)
img_snp_01_closeopen = open(close(img_snp_01, kernel), kernel)
img_snp_005_closeopen = open(close(img_snp_005, kernel), kernel)

# output
cv2.imwrite('imgs/gaussian_10.jpg', img_gaussian_10)
cv2.imwrite('imgs/gaussian_10_3box.jpg', img_gaussian_10_3box)
cv2.imwrite('imgs/gaussian_10_5box.jpg', img_gaussian_10_5box)
cv2.imwrite('imgs/gaussian_10_3med.jpg', img_gaussian_10_3med)
cv2.imwrite('imgs/gaussian_10_5med.jpg', img_gaussian_10_5med)
cv2.imwrite('imgs/gaussian_10_openclose.jpg', img_gaussian_10_openclose)
cv2.imwrite('imgs/gaussian_10_closeopen.jpg', img_gaussian_10_closeopen)

cv2.imwrite('imgs/gaussian_30.jpg', img_gaussian_30)
cv2.imwrite('imgs/gaussian_30_3box.jpg', img_gaussian_30_3box)
cv2.imwrite('imgs/gaussian_30_5box.jpg', img_gaussian_30_5box)
cv2.imwrite('imgs/gaussian_30_3med.jpg', img_gaussian_30_3med)
cv2.imwrite('imgs/gaussian_30_5med.jpg', img_gaussian_30_5med)
cv2.imwrite('imgs/gaussian_30_openclose.jpg', img_gaussian_30_openclose)
cv2.imwrite('imgs/gaussian_30_closeopen.jpg', img_gaussian_30_closeopen)

cv2.imwrite('imgs/snp_005.jpg', img_snp_005)
cv2.imwrite('imgs/snp_005_3box.jpg', img_snp_005_3box)
cv2.imwrite('imgs/snp_005_5box.jpg', img_snp_005_5box)
cv2.imwrite('imgs/snp_005_3med.jpg', img_snp_005_3med)
cv2.imwrite('imgs/snp_005_5med.jpg', img_snp_005_5med)
cv2.imwrite('imgs/snp_005_openclose.jpg', img_snp_005_openclose)
cv2.imwrite('imgs/snp_005_closeopen.jpg', img_snp_005_closeopen)

cv2.imwrite('imgs/snp_01.jpg', img_snp_01)
cv2.imwrite('imgs/snp_01_3box.jpg', img_snp_01_3box)
cv2.imwrite('imgs/snp_01_5box.jpg', img_snp_01_5box)
cv2.imwrite('imgs/snp_01_3med.jpg', img_snp_01_3med)
cv2.imwrite('imgs/snp_01_5med.jpg', img_snp_01_5med)
cv2.imwrite('imgs/snp_01_openclose.jpg', img_snp_01_openclose)
cv2.imwrite('imgs/snp_01_closeopen.jpg', img_snp_01_closeopen)

# Calculate SNR
img = img.astype("float")
img_gaussian_10 = img_gaussian_10.astype("float")
img_gaussian_10_3box = img_gaussian_10_3box.astype("float")
img_gaussian_10_5box = img_gaussian_10_5box.astype("float")
img_gaussian_10_3med = img_gaussian_10_3med.astype("float")
img_gaussian_10_5med = img_gaussian_10_5med.astype("float")
img_gaussian_10_openclose = img_gaussian_10_openclose.astype("float")
img_gaussian_10_closeopen = img_gaussian_10_closeopen.astype("float")

img_gaussian_30 = img_gaussian_30.astype("float")
img_gaussian_30_3box = img_gaussian_30_3box.astype("float")
img_gaussian_30_5box = img_gaussian_30_5box.astype("float")
img_gaussian_30_3med = img_gaussian_30_3med.astype("float")
img_gaussian_30_5med = img_gaussian_30_5med.astype("float")
img_gaussian_30_openclose = img_gaussian_30_openclose.astype("float")
img_gaussian_30_closeopen = img_gaussian_30_closeopen.astype("float")

img_snp_005 = img_snp_005.astype("float")
img_snp_005_3box = img_snp_005_3box.astype("float")
img_snp_005_5box = img_snp_005_5box.astype("float")
img_snp_005_3med = img_snp_005_3med.astype("float")
img_snp_005_5med = img_snp_005_5med.astype("float")
img_snp_005_openclose = img_snp_005_openclose.astype("float")
img_snp_005_closeopen = img_snp_005_closeopen.astype("float")

img_snp_01 = img_snp_01.astype("float")
img_snp_01_3box = img_snp_01_3box.astype("float")
img_snp_01_5box = img_snp_01_5box.astype("float")
img_snp_01_3med = img_snp_01_3med.astype("float")
img_snp_01_5med = img_snp_01_5med.astype("float")
img_snp_01_openclose = img_snp_01_openclose.astype("float")
img_snp_01_closeopen = img_snp_01_closeopen.astype("float")

#normalization
for i in range(h):
	for j in range(w):
		img[i][j] /= 255
		img_gaussian_10[i][j] /= 255
		img_gaussian_10_3box[i][j] /= 255
		img_gaussian_10_5box[i][j] /= 255
		img_gaussian_10_3med[i][j] /= 255
		img_gaussian_10_5med[i][j] /= 255
		img_gaussian_10_openclose[i][j] /= 255
		img_gaussian_10_closeopen[i][j] /= 255
		img_gaussian_30[i][j] /= 255
		img_gaussian_30_3box[i][j] /= 255
		img_gaussian_30_5box[i][j] /= 255
		img_gaussian_30_3med[i][j] /= 255
		img_gaussian_30_5med[i][j] /= 255
		img_gaussian_30_openclose[i][j] /= 255
		img_gaussian_30_closeopen[i][j] /= 255
		img_snp_005[i][j] /= 255
		img_snp_005_3box[i][j] /= 255
		img_snp_005_5box[i][j] /= 255
		img_snp_005_3med[i][j] /= 255
		img_snp_005_5med[i][j] /= 255
		img_snp_005_openclose[i][j] /= 255
		img_snp_005_closeopen[i][j] /= 255				
		img_snp_01[i][j] /= 255
		img_snp_01_3box[i][j] /= 255
		img_snp_01_5box[i][j] /= 255
		img_snp_01_3med[i][j] /= 255
		img_snp_01_5med[i][j] /= 255
		img_snp_01_openclose[i][j] /= 255
		img_snp_01_closeopen[i][j] /= 255


print('img_gaussian_10: {}'.format(SNR(img, img_gaussian_10)))
print('img_gaussian_10_3box: {}'.format(SNR(img, img_gaussian_10_3box)))
print('img_gaussian_10_5box: {}'.format(SNR(img, img_gaussian_10_5box)))
print('img_gaussian_10_3med: {}'.format(SNR(img, img_gaussian_10_3med)))
print('img_gaussian_10_5med: {}'.format(SNR(img, img_gaussian_10_5med)))
print('img_gaussian_10_openclose: {}'.format(SNR(img, img_gaussian_10_openclose)))
print('img_gaussian_10_closeopen: {}'.format(SNR(img, img_gaussian_10_closeopen)))

print('img_gaussian_30: {}'.format(SNR(img, img_gaussian_30)))
print('img_gaussian_30_3box: {}'.format(SNR(img, img_gaussian_30_3box)))
print('img_gaussian_30_5box: {}'.format(SNR(img, img_gaussian_30_5box)))
print('img_gaussian_30_3med: {}'.format(SNR(img, img_gaussian_30_3med)))
print('img_gaussian_30_5med: {}'.format(SNR(img, img_gaussian_30_5med)))
print('img_gaussian_30_openclose: {}'.format(SNR(img, img_gaussian_30_openclose)))
print('img_gaussian_30_closeopen: {}'.format(SNR(img, img_gaussian_30_closeopen)))

print('img_snp_005: {}'.format(SNR(img, img_snp_005)))
print('img_snp_005_3box: {}'.format(SNR(img, img_snp_005_3box)))
print('img_snp_005_5box: {}'.format(SNR(img, img_snp_005_5box)))
print('img_snp_005_3med: {}'.format(SNR(img, img_snp_005_3med)))
print('img_snp_005_5med: {}'.format(SNR(img, img_snp_005_5med)))
print('img_snp_005_openclose: {}'.format(SNR(img, img_snp_005_openclose)))
print('img_snp_005_closeopen: {}'.format(SNR(img, img_snp_005_closeopen)))

print('img_snp_01: {}'.format(SNR(img, img_snp_01)))
print('img_snp_01_3box: {}'.format(SNR(img, img_snp_01_3box)))
print('img_snp_01_5box: {}'.format(SNR(img, img_snp_01_5box)))
print('img_snp_01_3med: {}'.format(SNR(img, img_snp_01_3med)))
print('img_snp_01_5med: {}'.format(SNR(img, img_snp_01_5med)))
print('img_snp_01_openclose: {}'.format(SNR(img, img_snp_01_openclose)))
print('img_snp_01_closeopen: {}'.format(SNR(img, img_snp_01_closeopen)))
