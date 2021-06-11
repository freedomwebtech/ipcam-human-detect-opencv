import urllib.request
import time
import numpy as np
import cv2

url='http://192.168.0.102:8080/shot.jpg'

time.sleep(0.1)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
    # Use urllib to get the image from the IP camera
    imgResp = urllib.request.urlopen(url)
    
    # Numpy to convert into a array
    imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
    # Finally decode the array to OpenCV usable format ;) 
    image = cv2.imdecode(imgNp,-1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes, weights = hog.detectMultiScale(image, winStride=(8,8) )
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(image, (xA, yA), (xB, yB),(0, 255, 0), 2)
    img = cv2.resize(image,(1028,750))
    cv2.imshow("Frame", img);
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
       break
