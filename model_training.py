import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np
import time
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam

IMAGE_SIZE = 48

X = pickle.load(open("X-{}.pickle".format(IMAGE_SIZE), "rb"))
y = np.array(pickle.load(open("y-{}.pickle".format(IMAGE_SIZE), "rb")))
X = X/255.0

dense_layers = [1]
conv_layers =  [3]
layer_sizes =  [128]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "emo-analysis-Adam-decay-lr-10-epoch-{}-conv-{}-nodes-{}-dense-{}.model".format(conv_layer, layer_size, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

            model = Sequential()
 
            model.add(Conv2D(layer_size, kernel_size=(3, 3), activation='relu', input_shape=X.shape[1:]))
            model.add(Conv2D(layer_size*2, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            
            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size*4, (3, 3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.25))
            
            model.add(Flatten())
            
            for l in range(dense_layer):
                model.add(Dense(layer_size*4, activation='relu'))
                model.add(Dropout(0.5))  
                
            model.add(Dense(1024, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(7, activation='softmax'))

            model.compile(loss="sparse_categorical_crossentropy", # 7 categories to predict
                                     optimizer=Adam(lr=0.0001, decay=1e-6),  
                                     metrics=['accuracy'])

            model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1, callbacks=[tensorboard])

            model.save("models/"+NAME)