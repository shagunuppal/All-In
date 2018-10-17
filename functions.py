import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def read_gray_image(path):
	img = cv.imread(path)
	img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	return img, img_gray

def threshold_otsu_method(img_gray):
	ret,thresholded_img = cv.threshold(img_gray, 0, 255, cv.THRESH_OTSU)
	return thresholded_img

# canny method
def extract_contour(thresholded_img):
	img_edges = cv.Canny(thresholded_img,50,100)
	return img_edges

def prob_hough_transform(img, img_edges):
	minLineLength = 10
	maxLineGap = 20
	lines = cv.HoughLinesP(img_edges,1,np.pi/180,100,minLineLength,maxLineGap)
	print (len(lines))
	for n in range(len(lines)):
		for x1,y1,x2,y2 in lines[n]:
			cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)
	cv.imshow('img',img)
	cv.waitKey(0)
	cv.destroyAllWindows()

def hough_transform(img,img_edges):						
	lines = cv.HoughLines(img_edges,1,np.pi/180,150)
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
	cv.imshow('img',img)
	cv.waitKey(0)
	cv.destroyAllWindows()


if (__name__ == '__main__'):
	image, gray = read_gray_image('./card.jpg')
	thresholded = threshold_otsu_method(gray)
	edges = extract_contour(thresholded)
	hough_transform(image, edges)
	