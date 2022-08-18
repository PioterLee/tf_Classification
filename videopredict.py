
import numpy as np
from PIL import Image
import os
from os import listdir
from os.path import isfile, isdir, join
import tensorflow as tf
import cv2


predic_class = ''
model_load = tf.keras.models.load_model('best_img_model')

img_height = 180
img_width = 180

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_height, img_width))

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 

    predictions = model_load.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_names = ['00','01','02','03','04','05','06','07','08','09']
    predic_class = class_names[np.argmax(score)]

    cv2.imshow("Prediction", frame)
    key=cv2.waitKey(1)
    if key == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

def getpredict():
    return predic_class
