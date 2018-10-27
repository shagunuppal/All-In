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
	# plt.close()
	return img, img_gray

def threshold_otsu_method(img_gray):
	ret,thresholded_img = cv.threshold(img_gray, 0, 255, cv.THRESH_OTSU)
	plt.imshow(thresholded_img, cmap='gray', interpolation='nearest')
	plt.savefig('img_thresholded.jpg')
	plt.close()
	return thresholded_img

# canny method
def extract_contour(thresholded_img):
	THRESHOLD_MIN = 10
	THRESHOLD_MAX = 250
	img_edges = cv.Canny(thresholded_img, THRESHOLD_MIN, THRESHOLD_MAX)
	plt.imshow(img_edges, cmap='gray', interpolation='nearest')
	plt.savefig('img_contour.jpg')
	plt.close()
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
	plt.close()

def prob_hough_transform(img, img_edges):
	minLineLength = 40
	maxLineGap = 50
	lines = cv.HoughLinesP(img_edges,1,np.pi/180,100,minLineLength,maxLineGap)
	for n in range(len(lines)):
		for x1,y1,x2,y2 in lines[n]:
			cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)
	plt.imshow(img, cmap='gray', interpolation='nearest')
	plt.savefig('img_hough_prob.jpg')
	plt.close()

def hough_transform(img,img_edges):						
	lines = cv.HoughLines(img_edges,1,np.pi/180,75)
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
	plt.savefig('img_hough.jpg')
	plt.close()

def find_rectangle(thresholded_img, img):
	_, contours, hierarchy = cv.findContours(thresholded_img, 1, 2)
	cnt = contours[0]
	M = cv.moments(cnt)
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	epsilon = 0.1*cv.arcLength(cnt, True)
	approx = cv.approxPolyDP(cnt, epsilon, True)
	hull = cv.convexHull(cnt)
	rect = cv.minAreaRect(cnt)
	box = cv.boxPoints(rect)
	box = np.int0(box)
	im = cv.drawContours(img,[box],0,(0,0,255),2)
	plt.imshow(img)
	plt.savefig('img_with_rect.jpg')
	plt.close()

def determine_rank(extracted_card):
	#possible_ranks = ['./A_template.jpg', './2_template.jpg', './3_template.jpg', './4_template.jpg', './5_template.jpg', './6_template.jpg', './7_template.jpg', './8_template.jpg', './9_template.jpg', './10_template.jpg', './J_template.jpg''./Q_template.jpg', './K_template.jpg']
	possible_ranks = ['./2_template.png', './8_template.jpg']
	
	img_rgb = cv.imread(extracted_card)
	img_rgb = cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB)
	img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
	rank_ = 0
	for i in possible_ranks:
		rank_ = template_matching_rank(img_gray, img_rgb, i)
		if(rank_>0):
			rank = i.split('/')[1].split('_')[0]
			print('Rank of the card is : ', rank)
			break


def determine_suit(extracted_card):
	possible_suits = ['./heart_template2.jpg', './spade_template.jpg']
	img_rgb = cv.imread(extracted_card) # opencv reads image in BGR by default.
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
	if(color=='red'):
		suit = template_matching_suit(img_gray, img_rgb, possible_suits[0], color)
	elif(color=='black'):
		suit = template_matching_suit(img_gray, img_rgb, possible_suits[1], color)

	print('Color of the card is : ', color)
	print('Suit of the card is : ', suit)


def template_matching_suit(image, img_rgb, template_, color):
	# detect suit
	suit = 0
	template = cv.imread(template_,0)
	w, h = template.shape[::-1]
	res = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
	threshold = 0.7
	loc = np.where( res >= threshold)
	if(len(loc[0]) > 0 and color=='red'):
		suit = 'heart'
	elif(len(loc[0]) == 0 and color=='red'):
		suit = 'diamond'
	elif(len(loc[0]) > 0 and color=='black'):
		suit = 'spade'
	elif(len(loc[0]) == 0 and color=='black'):
		suit = 'club'

	for pt in zip(*loc[::-1]):
		cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
	plt.imshow(img_rgb)
	plt.savefig('img_with_suit.jpg')
	plt.close()

	return suit

def template_matching_rank(image, img_rgb, template_):
	# detect rank
	template = cv.imread(template_,0)
	w, h = template.shape[::-1]
	res = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
	threshold = 0.95
	loc = np.where( res >= threshold)
	for pt in zip(*loc[::-1]):
		cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
	plt.imshow(img_rgb)
	plt.savefig('img_with_rank.jpg')
	plt.close()

	return len(loc[0])


if (__name__ == '__main__'):
	image, gray = read_gray_image('./card.jpg')
	thresholded = threshold_otsu_method(gray)
	#find_rectangle(thresholded, image)
	edges = extract_contour(thresholded)
	#edges = contours(thresholded)
	#(image, edges)
	hough_transform(image, edges)
	#img_with_rect = find_rectangle(thresholded, image)
	determine_suit('./2.jpg')
	determine_rank('./2.jpg')

	