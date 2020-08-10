import cv2
from skimage.feature import local_binary_pattern
import numpy as np
from scipy.stats import itemfreq
from os import listdir
from os.path import isfile, join
import pandas as pd

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

def creating_descriptor(data_path, class_name, label):
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    Training_Data, Labels = [], []

    # Preparing the radius and the no. of points of the LBP descriptor
    radius = 3
    no_points = 8 * radius

    desc = LocalBinaryPatterns(no_points, radius)

    for i, files in enumerate(onlyfiles):
        # Getting image full path
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, 0)
        
        hist = desc.describe(images)
        
        # Appending the histogram to the training data
        Training_Data.append(np.asarray(hist, dtype="float64"))
        Labels.append(np.asarray(label, dtype="float64"))
        #Training_Data.append(hist)
        #Labels.append(label)

    #Labels = np.asarray(Labels, dtype=np.int32)
    #print("{0}|{1}".format(Training_Data, Labels))
    data_dict = { 'LBP': Training_Data, 'Labels': Labels }
    data_df = pd.DataFrame(data_dict)
    pd.to_pickle(data_df, data_path + 'pickle/' + class_name + '.pkl')


creating_descriptor('./dataset/A_gesture/', 'A_gesture', 0.0)
creating_descriptor('./dataset/B_gesture/', 'B_gesture', 1.0)
creating_descriptor('./dataset/C_gesture/', 'C_gesture', 2.0)
creating_descriptor('./dataset/nothing/', 'nothing', -1.0)