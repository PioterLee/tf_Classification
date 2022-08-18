
import numpy as np
import cv2

img_height = 180
img_width = 180

cap = cv2.VideoCapture(0)
count = 0
while True:
    _, frame = cap.read()

    img = cv2.resize(frame, (img_height, img_width))

    cv2.imshow("Prediction", frame)
    s_count = str(count).zfill(5)
    cv2.imwrite('imgfolder/'+s_count+'_.jpg',img)
    count+=1
    key=cv2.waitKey(60)
    if key == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

