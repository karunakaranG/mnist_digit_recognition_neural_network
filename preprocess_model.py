import cv2
import matplotlib.pyplot as plt
import numpy as np

def preprocessed_image(imagee):
    
    image=imagee.copy()
    #ret,thresh1 = cv2.threshold(image,100,255,cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(image,(5,5),0)
    ret3,thresh1 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    (_,cnts,_) = cv2.findContours(thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(cnts, key = cv2.contourArea, reverse = True)[:]
    cv2.drawContours(image,contours, -1, (0,255,0), 3)
    x,y,w,h = cv2.boundingRect(contours[0])
    roi=imagee[y:y+h,x:x+w]
    exact=cv2.resize(roi,(25,25))
    exact[exact<100]=0
    exact=exact/255
    final_extract=exact.reshape((625))
    return final_extract
