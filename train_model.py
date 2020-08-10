import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def preparing_data(data_path):
    data = pd.read_pickle(data_path)
    X = data['LBP'].to_numpy()
    y = data['Labels'].to_numpy()
    #print("{0}|{1}".format(type(X), type(y)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train.reshape(-1, 1), X_test.reshape(-1, 1), y_train, y_test
    
def train_model(model, X_train, y_train, bash):
    print('Training ' + bash)
    print("X shape: {0}| y shape: {1}".format(X_train.shape, y_train.shape))

    return model.fit(X_train, y_train)


bashA_train_X, bashA_test_X, bashA_train_y, bashA_test_y = preparing_data('./dataset/A_gesture/pickle/A_gesture.pkl')
bashB_train_X, bashB_test_X, bashB_train_y, bashB_test_y = preparing_data('./dataset/B_gesture/pickle/B_gesture.pkl')
bashC_train_X, bashC_test_X, bashC_train_y, bashC_test_y = preparing_data('./dataset/C_gesture/pickle/C_gesture.pkl')
bashnot_train_X, bashnot_test_X, bashnot_train_y, bashnot_test_y = preparing_data('./dataset/nothing/pickle/nothing.pkl')
print(type(bashA_train_X[0]))
print(type(bashA_train_y[0]))
print(bashA_train_X[0])
print(bashA_train_y[0])
#print("{0}|{1}".format(bashA_train_X.shape, bashA_train_y.shape))

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf_ba = train_model(clf, bashA_train_X, bashA_train_y, 'Bash A')




