
import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import tensorflow as tf
import cv2
from PIL import Image

mypath = 'intput_img/'
outputpath = 'output_img/'
files = listdir(mypath)


model_load = tf.keras.models.load_model('best_img_model')


img_height = 180
img_width = 180
for file in files:
    imgpath = join(mypath, file)
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_height, img_width))
    # img = img.astype(np.float32)


    # imgpath = join(mypath, file)
    # img = tf.keras.utils.load_img(
    #     imgpath, target_size=(img_height, img_width)
    # )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # tf.keras模型經過優化，可以一次對一批或一組示例進行預測。因此，即使您使用的是單個圖像，也需要將其添加到列表中


    predictions = model_load.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_names = ['00','01','02','03','04','05','06','07','08','09']
    # print(
    #     "class : {} ,score : {:.2f} %."
    #     .format(class_names[np.argmax(score)], 100 * np.max(score))
    # )
    
    cv2.imwrite(outputpath+str(class_names[np.argmax(score)])+'_'+file+'_.jpg',cv2.imread(imgpath))