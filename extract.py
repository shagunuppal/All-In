import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from scipy.ndimage import label

def extract(img):
	cards = []
	found = 0
	w = 640
	gamma = 2
	window = 0.8
	angle = 5
	min_line_length = 20

	scaling_factor = w / img.shape[1]
	img_res = cv.resize(img, None, fx = scaling_factor, fy = scaling_factor, interpolation = cv.INTER_CUBIC)

	img_gray = cv.cvtColor(img_res, cv.COLOR_BGR2GRAY)
	h = img.shape[0]

	min_size = 10 * math.sqrt(w)
	img_ctr = np.power(img_gray, gamma)

	maxValue = 0.8 * np.amax(img_ctr)
	ret, img_bw = cv.threshold(img_ctr, maxValue, 255, cv.THRESH_BINARY)
	L, numL = label(img_bw)

	for l in range(numL):
		arr = np.zeros(L.shape)
		for i in range(L.shape[0]):
			for j in range(L.shape[1]):
				if (L[i][j] == l):
					arr[i][j] = 255
		radius_disk = round(math.sqrt(np.count_nonzero(arr)/2))
		SE = cv.getStructuringElement(cv.MORPH_ELLIPSE, (radius_disk, radius_disk))

		img_closed = cv.dilate(cv.erode(arr, kernel = SE, iterations = 1), kernel = SE, iterations = 1)
		
		if (np.count_nonzero(img_closed) > min_size):
			y = []
			x = []
			for i in range(img_closed.shape[0]):
				for j in range(img_closed.shape[1]):
					if (img_closed[i][j] > 0):
						y.append(i)
						x.append(j)

			if (((abs(np.mean(y) - h / 2)) < h * window / 2) and 
				((abs(np.mean(x) - w / 2)) < w * window / 2)
				):
				borders = cv.dilate(img_closed, kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 1)), iterations = 1) - img_closed
				lines = cv.HoughLines(borders,1,np.pi/180,150)		
			if(l==0):
				break
			edges = cv2.Canny(borders,50,150,apertureSize = 3)
			lines = cv2.HoughLines(edges,1,np.pi/180,200)
			max_len = 0
			pts = np.zeros(4,2)
			for n in range(lines.shape[1]):
				line1 = lines(n)
				for m in range(n+1, lines.shape[1]):
					if(n!=m):
						line2 = lines(m)
						if(abs(line1[0][1]-line2[0][1]) < angle and abs(line1[0][0] - line2[0][0]) > min_line_length or norm(line2.point1 - line2.point2) > max_len):
							pts = [line1.point1, line1.point2, line2.point2, line2.point1]

		plt.imshow(lines, cmap='gray', interpolation='nearest')
		plt.savefig('img.jpg')
		print('hello')


if (__name__ == '__main__'):
	img = cv.imread('./card.jpg')
	extract(img)

































