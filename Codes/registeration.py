import cv2 as cv 
import numpy as np

num_features = 100
filename = './poker.jpg'
filename_card = './card.jpg'

img = cv.imread(filename, cv.IMREAD_COLOR)
img_card = cv.imread(filename_card, cv.IMREAD_COLOR)
img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
img_gray_card = cv.cvtColor(img_card, cv.COLOR_RGB2GRAY)

orb = cv.ORB_create(num_features)

keypoints, descriptors = orb.detectAndCompute(img_gray, None)
keypoints_card, descriptors_card = orb.detectAndCompute(img_gray_card, None)

#matcher = cv.DescriptorMatcher_create()