import cv2
from skimage.feature import local_binary_pattern
import numpy as np
from scipy.stats import itemfreq
from os import listdir
from os.path import isfile, join
import csv

def creating_descriptor(data_path, class_name, label):
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    Training_Data, Labels = [], []

    for i, files in enumerate(onlyfiles):
        # Getting image full path
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, 0)
        
        # Preparing the radius and the no. of points of the LBP descriptor
        radius = 3
        no_points = 8 * radius

        # Computing the LBP descriptor
        lbp = local_binary_pattern(images, no_points, radius, method='uniform')

        # Calculate the histogram
        x = itemfreq(lbp.ravel())

        # Normalize the histogram
        hist = x[:, 1]/sum(x[:, 1])
        
        # Appending the histogram to the training data
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(label)

    Labels = np.asarray(Labels, dtype=np.int32)
    data_dict = { 'LBP': Training_Data, 'Labels': Labels }
    with open(data_path + 'csv/' + class_name + '.csv', 'w') as f:
        w = csv.DictWriter(f, data_dict.keys())
        w.writeheader()
        w.writerow(data_dict)


creating_descriptor('./dataset/A_gesture/', 'A_gesture', 0)
creating_descriptor('./dataset/B_gesture/', 'B_gesture', 1)
creating_descriptor('./dataset/C_gesture/', 'C_gesture', 2)
creating_descriptor('./dataset/nothing/', 'nothing', -1)