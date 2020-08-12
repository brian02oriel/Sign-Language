import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from os import listdir
from os.path import isfile, join
import pandas as pd
from joblib import load

class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius
	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))
		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
		# return the histogram of Local Binary Patterns
		return hist


cap = cv2.VideoCapture(0)

# This the ROI size, the size of images saved will be box_size -10
box_size = 240
    
# Getting the width of the frame from the camera properties
width = int(cap.get(3))

radius = 3
no_points = 8 * radius
desc = LocalBinaryPatterns(no_points, radius)
decoding = {
	0.: "A",
	1.: "B",
	2.: "C",
	3.: "D",
	4.: "E",
	5.: "F"
	}

while(True):
	_, img = cap.read()
	img = cv2.flip(img, 1)
	#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cv2.rectangle(img, (width - box_size, 0), (width, box_size), (0, 0, 255), 2)
	roi = img[5: box_size-5 , width-box_size + 5: width -5]
	roi = cv2.resize(roi, (200, 200))
	roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(roi)
	hist = np.array(hist)
	#print(hist, len(hist))
	model = load('classifier.joblib')
	results = model.predict(hist.reshape(1, -1))
	if(results[0] > -1):
		cv2.rectangle(img, (width - box_size, 0), (width, box_size), (0, 255, 0), 3)
		cv2.rectangle(img, (width - box_size - 2, box_size + 30), (width, box_size), (0, 255, 0), -1)
		cv2.putText(img, decoding[results[0]], (width - (round(box_size/2)), box_size + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
		#print(decoding[results[0]])

	cv2.imshow('Face Recognition', img)
	if(cv2.waitKey(1) & 0xFF == ord('q')):
		break

cap.release()
cv2.destroyAllWindows()