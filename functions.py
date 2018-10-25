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
	plt.savefig('img_contour.jpg')
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

def template_matching():
	img_rgb = cv.imread('./8.jpg') # opencv reads image in BGR by default.
	img_rgb = cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB) # convert to standard RGB for matplotlib deafult.
	
	# get rgb pixels color count
	red = img_rgb[:,:,0]
	green = img_rgb[:,:,1]
	blue = img_rgb[:,:,2] 

	red_count = np.sum(red)
	green_count = np.sum(green)
	blue_count = np.sum(blue)

	# detecting the color of the card
	if(red_count>green_count):
		color = 'red'
	else:
		color = 'black'

	img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY) # convert to grayscale

	# detect suit 
	if(color=='red'):
		template = cv.imread('./heart_template.jpg',0)
		w, h = template.shape[::-1]
		res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
		threshold = 0.95
		loc = np.where( res >= threshold)
		if(len(loc)>0):
			suit = 'heart'
		else:
			suit = 'diamond'
	else:
		template = cv.imread('./spade_template.jpg',0)
		w, h = template.shape[::-1]
		res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
		threshold = 0.95
		loc = np.where( res >= threshold)
		if(len(loc)>0):
			suit = 'spade'
		else:
			suit = 'club'

	# detect rank


	for pt in zip(*loc[::-1]):
		cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
	plt.imshow(img_rgb)
	plt.savefig('img4.jpg')

if (__name__ == '__main__'):
	#template_matching()
	image, gray = read_gray_image('./king.jpg')
	thresholded = threshold_otsu_method(gray)
	edges = extract_contour(thresholded)
	#edges = contours(thresholded)
	#prob_hough_transform(image, edges)
	#hough_transform(image, edges)


	