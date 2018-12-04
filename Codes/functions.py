import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def read_gray_image(path):
	img = cv.imread(path)
	img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	plt.imshow(img_gray, cmap='gray', interpolation='nearest')
	plt.savefig('./results/img_gray.png')
	plt.close()
	return img, img_gray

def threshold_otsu_method(img_gray):
	ret,thresholded_img = cv.threshold(img_gray, 0, 255, cv.THRESH_OTSU)
	plt.imshow(thresholded_img, cmap='gray', interpolation='nearest')
	plt.savefig('./results/img_thresholded.png')
	plt.close()
	return thresholded_img

# canny method
def extract_contour(thresholded_img):
	THRESHOLD_MIN = 3
	THRESHOLD_MAX = 30
	img_edges = cv.Canny(thresholded_img, THRESHOLD_MIN, THRESHOLD_MAX)
	plt.imshow(img_edges, cmap='gray', interpolation='nearest')
	plt.savefig('./results/img_contour.png')
	plt.close()
	return img_edges

def prob_hough_transform(img, img_edges):
	minLineLength = 10
	maxLineGap = 35
	lines = cv.HoughLinesP(img_edges,1,np.pi/180,1,minLineLength,maxLineGap)
	img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	for n in range(len(lines)):
		for x1,y1,x2,y2 in lines[n]:
			cv.line(img_rgb,(x1,y1),(x2,y2),(0,255,0),2)
	#plt.imshow(img_rgb, cmap='gray', interpolation='nearest')
	#plt.savefig('./results/img_hough_prob.png')
	#plt.close()

def hough_transform(img,img_edges):	
	line_array = []
	lines = cv.HoughLines(img_edges,1,np.pi/2,90)
	img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	indexes = [0]*len(lines)
	# to remove lines in extreme proximity, maybe due to the design of the card.
	# ========================================= preprocessing 
	for r1 in range(len(lines)):
		if(indexes[r1]==0):
			r = lines[r1]
			for v1 in range(len(lines)):
				if(indexes[v1]==0):
					v = lines[v1]
					if(v1!=r1):
						if(abs(r[0][0]-v[0][0])<5 and r[0][1]-v[0][1]==0):
							if(r[0][0]>v[0][0]):
								indexes[r1] = 1
							else:
								indexes[v1] = 1
	lines1 = []
	#print('Length:', len(lines))
	for i in range(len(indexes)):
		if(indexes[i]==0):
			lines1.append(lines[i])
	lines = np.asarray(lines1)
	#print(len(lines))
	x = img.shape[0]
	y = img.shape[1]
	for n in range(len(lines)):	
		for rho,theta in lines[n]:
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a*rho
			y0 = b*rho
			x1 = round(x0 + y*(-b))
			y1 = round(y0 + x*(a))
			x2 = round(x0 - y*(-b))
			y2 = round(y0 - x*(a))
			line_array.append([[x1,y1],[x2,y2]])
			cv.line(img_rgb,((int)(x1),(int)(y1)),((int)(x2),(int)(y2)),(0,0,255),2)
	plt.imshow(img_rgb, cmap='gray', interpolation='nearest')
	plt.savefig('./results/img_hough.png')
	plt.close()
	return line_array

def find_corners(img, lines):
	# equation of a line
	points = []
	points_1 = []
	points_2 = []
	for i in lines:
		x1, y1 = i[0][0], i[0][1]
		x2, y2 = i[1][0], i[1][1]
		for j in lines:
			if(i!=j):
				x11, y11 = j[0][0], j[0][1]
				x12, y12 = j[1][0], j[1][1]
				if(x1!=x2 and x11!=x12):
					m1 = ((float)(y2-y1)/(float)((x2-x1)))
					c1 = y1 - x1*(m1)
					m2 = ((float)(y12-y11)/(float)((x12-x11)))
					c2 = y11 - x11*(m2)
					if(m1!=m2):
						point_y = (c1*m2 - c2*m1) / (m2-m1)
						if(m1!=0):
							point_x = (point_y - c1) / m1
						else:
							point_x = (point_y - c2) / m2
						points.append([point_x, point_y])
				elif(x1!=x2):
					m1 = ((float)(y2-y1)/(float)((x2-x1)))
					c1 = y1 - x1*(m1)
					point_x = x11
					point_y = m1*point_x + c1
					points.append([point_x, point_y])
					points_1.append([point_x, point_y])
				elif(x11!=x12):
					m2 = ((float)(y12-y11)/(float)((x12-x11)))
					c2 = y11 - x11*(m2)
					point_x = x1
					point_y = m2*point_x + c2
					points.append([point_x, point_y])
					points_2.append([point_x, point_y])
					
	points1 = []
	pts_1 = []
	pts_2 = []
	for i in points:
		if i not in points1:
			points1.append(i)
	for i in points_1:
		if i not in pts_1:
			pts_1.append(i)
	for i in points_2:
		if i not in pts_2:
			pts_2.append(i)
	return points1, pts_1, pts_2

def plot_points(img, points, name):
	green = [0, 255, 0]
	img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	x, y = img_rgb.shape[0], img.shape[1]
	#print('points:', points)
	for i in points:
		if(i[0]<y and i[1]<x):
			cv.circle(img_rgb,(int(i[0]),int(i[1])),5,(50,0,250),-1) 
	plt.imshow(img_rgb, cmap='gray', interpolation='nearest')
	plt.savefig('./results/'+name+'.png')
	plt.close()

def determine_rank(extracted_card):
	#possible_ranks = ['./A_template.png', './2_template.png', './3_template.png', './4_template.png', './5_template.png', './6_template.png', './7_template.png', './8_template.png', './9_template.png', './10_template.png', './J_template.png''./Q_template.png', './K_template.png']
	possible_ranks = ['./templates/ranks/2_black.png', './templates/ranks/3_red.png', './templates/ranks/3_red1.jpeg', './templates/ranks/7_red.png', './templates/ranks/8_red.jpg', './templates/ranks/9_black.png', './templates/ranks/10_red.png', './templates/ranks/A_red.png', './templates/ranks/4_black.png', './templates/ranks/7_red1.png', './templates/ranks/K_red.png', './templates/ranks/8_black.png', './templates/ranks/A_black1.png']
	img_rgb = cv.imread(extracted_card)
	img_rgb = cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB)
	img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
	rank_ = 0
	rank_ = template_matching_rank(img_gray, img_rgb, possible_ranks)
	#for i in possible_ranks:
		#rank_ = template_matching_rank(img_gray, img_rgb, i)
		#if(rank_>0):
	#print("=============================================================", rank_)
	rank = rank_.split('/')[3].split('_')[0]
	print('Rank of the card is : ', rank)
	return rank
	
def determine_suit(extracted_card):
	red_possible_suits = ['./templates/suits/heart/heart.png', './templates/suits/heart/heart1.png', './templates/suits/heart/heart2.png', './templates/suits/heart/heart3.png']
	black_possible_suits = ['./templates/suits/spade/spade.jpg', './templates/suits/spade/spade1.jpg', './templates/suits/spade/spade2.png', './templates/suits/spade/spade3.png', './templates/suits/spade/spade4.png']
	img_rgb = cv.imread(extracted_card) # opencv reads image in BGR by default.
	img_rgb = cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB) # convert to standard RGB for matplotlib deafult.
	# get rgb pixels color count
	red = img_rgb[:,:,0]
	#red = red[10:, 10:]
	#print("size", red.shape)
	green = img_rgb[:,:,1]
	blue = img_rgb[:,:,2] 
	red_count = np.sum(red)
	green_count = np.sum(green)
	blue_count = np.sum(blue)
	#print("Count : ", red_count, green_count, blue_count)
	# detecting the color of the card
	if(red_count>green_count):
		color = 'red'
	else:
		color = 'black'
	#color = 'black'
	img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY) # convert to grayscale
	if(color=='red'):
		suit = template_matching_suit(img_gray, img_rgb, red_possible_suits, color)
	elif(color=='black'):
		suit = template_matching_suit(img_gray, img_rgb, black_possible_suits, color)

	print('Color of the card is : ', color)
	print('Suit of the card is : ', suit)
	return suit


def template_matching_suit(image, img_rgb, templates, color):
	# detect suit
	max_length = 0
	#print(template)
	for template_ in templates:
		suit = 0
		#print(template_)
		template = cv.imread(template_,0)
		#print(template.shape)
		#print(template)
		w, h = template.shape[::-1]
		res = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
		threshold = 0.8
		loc1 = np.where( res >= threshold)
		if(len(loc1[0])>max_length):
			max_length = len(loc1[0])
			loc = loc1
	if(max_length > 0 and color=='red'):
		suit = 'heart'
	elif(max_length == 0 and color=='red'):
		suit = 'diamond'
	elif(max_length > 0 and color=='black'):
		suit = 'spade'
	elif(max_length == 0 and color=='black'):
		suit = 'club'

	if(max_length!=0):
		for pt in zip(*loc[::-1]):
			cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
		#plt.imshow(img_rgb)
		#plt.savefig('img_with_suit.png')
		#plt.close()
	return suit[0]

def template_matching_rank(image, img_rgb, templates):
	# detect rank
	max_length = 0
	name = ""
	for template_ in templates:
		rank = 0
		template = cv.imread(template_,0)
		w, h = template.shape[::-1]
		res = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
		threshold = 0.95
		loc1 = np.where( res >= threshold)
		if(len(loc1[0])>max_length):
			max_length = len(loc1[0])
			loc = loc1
			name = template_
	if(max_length!=0):
		for pt in zip(*loc[::-1]):
			cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
		#plt.imshow(img_rgb)
		#plt.savefig('img_with_rank.png')
		#plt.close()
	return name

def perspective_transform(img,points):
	# segregate points
	y = {}
	y1 = []
	y2 = []
	for i in points:
		if(i[1] not in y.keys()):
			y[i[1]] = [i]
		else:
			y[i[1]].append(i)
	k = list(y.keys())
	k.sort()
	y1 = y[k[0]]
	y1.sort(key=lambda x: x[0])
	y2 = y[k[1]]
	y2.sort(key=lambda x: x[0])

	index = []

	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	if(len(y1)%2==0):
		loop_increment = 2
	else:
		loop_increment = 1 
	#print('y1:', y1)
	#print('y2:', y2)
	for i in range(0, (int)(len(y1)), loop_increment):
		if(i<len(y1) and i+1<len(y1)):
			pts1 = np.float32([y1[i], y2[i], y1[i+1], y2[i+1]])
			pts2 = np.float32([[0,0],[0, 450],[450, 0],[450,450]])
			M = cv.getPerspectiveTransform(pts1,pts2)
			dst = cv.warpPerspective(img,M,(450,450))
			plt.imshow(dst, cmap='gray', interpolation='nearest')
			plt.savefig('./results/perspective/'+(str)(i)+'.png')
			index.append((str)(i))
			plt.close()
	return index

if (__name__ == '__main__'):
	i = './test_cards/card21.jpeg'
	image, gray = read_gray_image(i)
	thresholded = threshold_otsu_method(gray)
	edges = extract_contour(thresholded)
	lines = hough_transform(image, edges)
	points, pts_1, pts_2 = find_corners(image, lines)
	plot_points(image, points, 'points')
	index = perspective_transform(image, points)
	suit = []
	rank = []
	for i in index:
		s = determine_suit('./results/perspective/'+i+'.png')
		r = determine_rank('./results/perspective/'+i+'.png')
		suit.append(s)
		rank.append(r)
	cards = []
	for i in range(len(suit)):
		cards.append(rank[i] + suit[i])
	print ('CARDS are : ', cards)
