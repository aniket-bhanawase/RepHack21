import numpy as np
import cv2
import time as t
import tensorflow as tf

IMG_SIZE = 48

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

MODEL_DIR = "./best/"
STRECH = 10

CATEGORIES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

COLORS = {
     'angry':  (0, 0, 255), 
     'disgust': (0, 255, 255), 
     'fear':    (255,145,0), 
     'happy':   (0, 255, 0), 
     'neutral': (0, 53, 255), 
     'sad':     (255, 26, 0), 
     'surprise':(255, 0, 255)
}

MODEL_DIR = "./best/"

MODEL_NAME = "emo-analysis-Adam-decay-lr-3-conv-128-nodes-1-dense-1613837977.model"

model = tf.keras.models.load_model(MODEL_DIR + MODEL_NAME)

window_name = 'Image'
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0) 
thickness = 2
emotions = []
queue = []
while 1:
    try:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        time_str = str(int(t.time()))
        for (x,y,w,h) in faces:
            x-=STRECH
            y-=STRECH
            w+=STRECH*2
            h+=STRECH*2
            roi_gray = gray[y:y+h, x:x+w]
    #         roi_color = img[y:y+h, x:x+w]
            img_array = cv2.resize(roi_gray, (IMG_SIZE, IMG_SIZE))
            img_array = img_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
            org = (x-10, y-10)
            pred = model.predict([img_array])
            emotion = CATEGORIES[np.argmax(pred)]
            emotions.append(emotion)
            cv2.rectangle(img,(x,y),(x+w,y+h),COLORS[emotion],2)
            img=cv2.putText(img,emotion, org, font,
                           fontScale, COLORS[emotion], thickness, cv2.LINE_AA)

        cv2.imshow('img',img)
        if emotions:
            cv2.imwrite(r'./faces/'+'face-'+','.join(emotions)+str(int(t.time()))+'.png', img)
        emotions.clear()
    except Exception as e:
        pass
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()