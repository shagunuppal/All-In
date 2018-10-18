import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def read_gray_image(path):
	img = cv.imread(path)
	img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	# plt.imshow(img_gray, cmap='gray', interpolation='nearest')
	# plt.savefig('img.jpg')
	return img, img_gray

def threshold_otsu_method(img_gray):
	ret,thresholded_img = cv.threshold(img_gray, 0, 255, cv.THRESH_OTSU)
	# plt.imshow(thresholded_img, cmap='gray', interpolation='nearest')
	# plt.savefig('img.jpg')
	return thresholded_img

# canny method
def extract_contour(thresholded_img):
	img_edges = cv.Canny(thresholded_img,20,120, apertureSize=3)
	plt.imshow(img_edges, cmap='gray', interpolation='nearest')
	plt.savefig('img.jpg')
	return img_edges
	
def contours(im):
	BLACK_THRESHOLD = 200
	THIN_THRESHOLD = 10
	_, contours, hierarchy = cv.findContours(im, 1, 3)
	idx = 0
	for cnt in contours:
		idx += 1
		x, y, w, h = cv.boundingRect(cnt)
		roi = im[y:y + h, x:x + w]
		if h < THIN_THRESHOLD or w < THIN_THRESHOLD:
			continue
		cv.imwrite(str(idx) + '.png', roi)
		cv.rectangle(im, (x, y), (x + w, y + h), (200, 0, 0), 2)
	plt.imshow(im, cmap='gray', interpolation='nearest')
	plt.savefig('img.jpg')

def prob_hough_transform(img, img_edges):
	minLineLength = 40
	maxLineGap = 50
	lines = cv.HoughLinesP(img_edges,1,np.pi/180,100,minLineLength,maxLineGap)
	print (len(lines))
	for n in range(len(lines)):
		for x1,y1,x2,y2 in lines[n]:
			cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)
	plt.imshow(img, cmap='gray', interpolation='nearest')
	plt.savefig('img.jpg')

def hough_transform(img,img_edges):						
	lines = cv.HoughLines(img_edges,1,np.pi/180,250)
	print (len(lines))
	for n in range(len(lines)):	
		for rho,theta in lines[n]:
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a*rho
			y0 = b*rho
			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))
			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))
			cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)
	plt.imshow(img, cmap='gray', interpolation='nearest')
	plt.savefig('img.jpg')

if (__name__ == '__main__'):
	image, gray = read_gray_image('./2.jpg')
	thresholded = threshold_otsu_method(gray)
	#edges = extract_contour(thresholded)
	edges = contours(thresholded)
	#prob_hough_transform(image, edges)
	#hough_transform(image, edges)
	