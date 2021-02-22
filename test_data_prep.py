import random 
import pickle
import os
import cv2 
import numpy as np

TEST_DATA_DIR = "./data/test" # 48x48 px image
CATEGORIES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

testing_data = []

def create_testing_data():
    for CATEGORY in CATEGORIES:
        path = os.path.join(TEST_DATA_DIR, CATEGORY)
        for img_name in os.listdir(path):
            class_ = CATEGORIES.index(CATEGORY)
            img_array = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)
            testing_data.append([img_array, class_])
            
create_testing_data()

random.shuffle(testing_data)

IMG_SIZE = 48

X_test=[]
y_test=[]

for features, label in testing_data:
    X_test.append(features)
    y_test.append(label)
    
X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #1 for gray scale, 3 for color img0
y_test = np.array(y_test)


pickle_out=open("X-test-{}.pickle".format(IMG_SIZE), "wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()

pickle_out=open("y-test-{}.pickle".format(IMG_SIZE), "wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()