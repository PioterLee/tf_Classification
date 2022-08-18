import numpy as np
from flask import Flask,request,jsonify
from flask_cors import CORS
import threading
from PIL import Image
from os import listdir
from os.path import isfile, isdir, join
import tensorflow as tf
import cv2


app = Flask(__name__)
CORS(app)

# @app.route('/')
# def index():
#     return 'hello'


@app.route('/predict',methods=['GET'])
def getInput():
    classname = request.args.get('classname')
    if str(predic_class)==classname:
        re = 'True'
    else:
        re = 'False'
    return re

model_load = tf.keras.models.load_model('best_img_model')
predic_class = ''
def webcampredict(): # 要被執行的方法(函數)
    global predic_class
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
        cv2.putText(frame, predic_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Prediction", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    t = threading.Thread(target=webcampredict) 
    t.start() # 開始
    app.run(host='192.168.50.134',port = 3000,debug=False)
    
