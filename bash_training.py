import cv2
from skimage.feature import local_binary_pattern
import numpy as np
from scipy.stats import itemfreq
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score
from joblib import dump
    
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
        #Training_Data.append(np.asarray(hist, dtype="float64"))
        #Labels.append(np.asarray(label, dtype="float64"))
        Training_Data.append(hist)
        Labels.append(label)

    #Labels = np.asarray(Labels, dtype=np.int32)
    #print("{0}|{1}".format(Training_Data, Labels))
    #data_dict = { 'LBP': Training_Data, 'Labels': Labels }
    #data_df = pd.DataFrame(data_dict)
    #pd.to_pickle(data_df, data_path + 'pickle/' + class_name + '.pkl')
    return Training_Data, Labels

def measuringPerformance(name, model, X, y, y_true, y_pred):    
    print("Measuring performance of: ", name)
    
    # Cross validation score
    print("Cross validation score: ")
    print(cross_val_score(model, X, y, scoring="accuracy"))
    
    # Confusion matrix
    y_train_predict = cross_val_predict(model, X, y, cv=3)
    print("Confusion matrix: ")
    print(confusion_matrix(y, y_train_predict))
    
    # Accuracy score
    print("Accuracy: ")
    print(accuracy_score(y_true, y_pred))


Xba, yba = creating_descriptor('./dataset/A_gesture/', 'A_gesture', 0.0)
Xbb, ybb = creating_descriptor('./dataset/B_gesture/', 'B_gesture', 1.0)
Xbc, ybc = creating_descriptor('./dataset/C_gesture/', 'C_gesture', 2.0)
Xbd, ybd = creating_descriptor('./dataset/D_gesture/', 'D_gesture', 3.0)
Xbe, ybe = creating_descriptor('./dataset/E_gesture/', 'E_gesture', 4.0)
Xbf, ybf = creating_descriptor('./dataset/F_gesture/', 'F_gesture', 5.0)
Xnothing, ynothing = creating_descriptor('./dataset/nothing/', 'nothing', -1.0)

clf = KNeighborsClassifier(n_neighbors=5)
X = Xba + Xbb + Xbc + Xbd + Xbe + Xbf + Xnothing
y = yba + ybb + ybc + ybd + ybe + ybf + ynothing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(len(X_train), len(y_train))

bash1_X = [X_train[i] for i in range(0, round(len(X_train) * 0.25 ))]
bash1_y = [y_train[i] for i in range(0, round(len(y_train) * 0.25 ))]
print(len(bash1_X), len(bash1_y))

bash2_X = [X_train[i] for i in range(round(len(X_train) * 0.25 ),  round(len(X_train) * 0.5 ))]
bash2_y = [y_train[i] for i in range(round(len(X_train) * 0.25 ),  round(len(X_train) * 0.5 ))]
print(len(bash2_X), len(bash2_y))

bash3_X = [X_train[i] for i in range(round(len(X_train) * 0.5 ),  round(len(X_train) * 0.75 ))]
bash3_y = [y_train[i] for i in range(round(len(X_train) * 0.5 ),  round(len(X_train) * 0.75 ))]
print(len(bash3_X), len(bash3_y))

bash4_X = [X_train[i] for i in range(round(len(X_train) * 0.75 ),  len(X_train) )]
bash4_y = [y_train[i] for i in range(round(len(X_train) * 0.75 ),  len(X_train) )]
print(len(bash4_X), len(bash4_y))

clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
model1 = clf.fit(bash1_X, bash1_y)
result = model1.predict(X_test)
measuringPerformance("KNN BASH-1", model1, bash1_X, bash1_y, y_test, result)
print("\n")
#print("first bash: \n", result)

model2 = model1.fit(bash2_X, bash2_y)
result = model2.predict(X_test)
measuringPerformance("KNN BASH-2", model2, bash2_X, bash2_y, y_test, result)
print("\n")
#print("second bash: \n", result)

model3 = model2.fit(bash3_X, bash3_y)
result = model3.predict(X_test)
measuringPerformance("KNN BASH-3", model3, bash3_X, bash3_y, y_test, result)
print("\n")
#print("third bash: \n", result)

model4 = model3.fit(bash4_X, bash4_y)
result = model4.predict(X_test)
measuringPerformance("KNN BASH-4", model4, bash4_X, bash4_y, y_test, result)
print("\n")


dump(model4, 'classifier.joblib')




