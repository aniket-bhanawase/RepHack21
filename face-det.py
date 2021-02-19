import numpy as np
import cv2
import time as t

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('big_buck_bunny_720p_5mb.mp4')
cnt = 0
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    time_str = str(int(t.time()))
    for (x,y,w,h) in faces:
        print("x: {} y: {}, w: {}, h: {}".format(x,y,w,h))
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cv2.imwrite(r'/Users/aniketb/Documents/learning-workspace/face_detection/face_detection_data_collection/data/face_'+time_str+'.png', roi_color)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    cnt+=1
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()