import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random 
import pickle
from tqdm.notebook import tqdm

DATA_DIR = "./data/train" # 48x48 px image
CATEGORIES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

training_data = []

def create_training_data():
    with tqdm(total=len(CATEGORIES)) as pbar_cat:
        for CATEGORY in CATEGORIES:
            path = os.path.join(DATA_DIR, CATEGORY)
            imgs_list = os.listdir(path)
            with tqdm(total=len(imgs_list)) as pbar_img:
                for img_name in os.listdir(path):
                    class_ = CATEGORIES.index(CATEGORY)
                    img_array = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)
                    training_data.append([img_array, class_])
                    pbar_img.update(1)
            pbar_cat.update(1)
            
create_training_data()


random.shuffle(training_data)

IMG_SIZE = 48

X=[]
y=[]

with tqdm(total=len(CATEGORIES)) as pbar_t_data:
    for features, label in training_data:
        X.append(features)
        y.append(label)
        pbar_t_data.update(1)


X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #1 for gray scale, 3 for color img0
y = np.array(y)

 # expoet cleaned data

pickle_out=open("X-{}.pickle".format(IMG_SIZE), "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out=open("y-{}.pickle".format(IMG_SIZE), "wb")
pickle.dump(y, pickle_out)
pickle_out.close()