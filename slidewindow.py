import cv2
import matplotlib.pyplot as plt
import numpy as np

def find_u_r_contours(image):
        image[image>200]=255
        #gray1 = cv2.bilateralFilter(image, 11, 17, 17)
        #edged = cv2.Canny(gray1, 10,100)
        blur = cv2.GaussianBlur(image,(5,5),0)
        ret3,thresh1 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        value=[]
        (_,cnts,_) = cv2.findContours(thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = sorted(cnts, key = cv2.contourArea, reverse = True)[:]

        for c in contours:
        
                if len(c)>450 and len(c)<1000:
                        x,y,w,h = cv2.boundingRect(c)
                        roi=np.array(image[y:y+h,x:x+w])
                        roi=255-roi
                        roi[roi<150]=0
                        roii=cv2.resize(roi,(25,25))
                        plt.imshow(roii,cmap='gray')
                        plt.show()
                        value.append(roii)
        return value

def sliding_window(image,windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0],600):
		for x in range(0, image.shape[1],300):
			# yield the current window
			print("x value:",x)
			print("y value:",y)
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def main_slide(image1):
        #image = cv2.imread('copr.jpg',0)
        image=image1
        (winW, winH) = (450,500)
        final_val=[]
        for (x, y, window) in sliding_window(image,windowSize=(winW, winH)):
                clone = image.copy()
                immr=clone[y:y+winH,x:x+winW]
                imgr=np.array(immr,dtype=np.uint8)
                ing=find_u_r_contours(imgr)
                for cr in ing:
                        res=cr.reshape((625))
                        final_val.append((res/255))
                        
        return final_val





        #cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        #cv2.namedWindow('Window', cv2.WINDOW_NORMAL)
        #cv2.imshow("Window", clone)
        #cv2.waitKey(0)
        
"""

image = cv2.imread('copr.jpg',0)
image[image>200]=255
#gray1 = cv2.bilateralFilter(image, 11, 17, 17)
#edged = cv2.Canny(gray1, 10,100)
blur = cv2.GaussianBlur(image,(5,5),0)
ret3,thresh1 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.imshow(thresh1,cmap='gray')
plt.show()
(_,cnts,_) = cv2.findContours(thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours = sorted(cnts, key = cv2.contourArea, reverse = True)[:]

for c in contours:
        
        if len(c)<450:
                x,y,w,h = cv2.boundingRect(c)
                roi=np.array(image[y:y+h,x:x+w])
                roi[roi>200]=255
                roi=255-roi
                roii=cv2.resize(roi,(25,25))
                plt.imshow(roii,cmap='gray')
                plt.show()

"""
        
"""
        if len(c)>450 and len(c)<2000:
                x,y,w,h = cv2.boundingRect(c)
                roi=np.array(image[y:y+h,x:x+w])
                roi[roi>200]=255
                roi=255-roi
                roii=cv2.resize(roi,(25,25))
                plt.imshow(roii,cmap='gray')
                plt.show()
"""

        
    
#cv2.drawContours(image, contours, -1, (0,255,0), 3)
#cv2.imshow("contour image", image)
#cv2.waitKey(0)
#plt.imshow(image,cmap='gray')
#plt.show()

"""
gray = cv2.bilateralFilter(image, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)

(_,cnts,_) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(cnts, key = cv2.contourArea, reverse = True)[:]
screenCnt = None
#cv2.drawContours(image, contours, -1, (0,255,0), 3)
#cv2.drawContours(image, contours, 3, (0,255,0), 3)
#for c in contours:
#    cnt = contours[c]
cv2.drawContours(image,contours, -1, (0,255,0), 3)
cv2.imshow("Game Boy Screen", image)
cv2.waitKey(0)
print(len(contours))

#x,y,w,h = cv2.boundingRect(cnt)
#roi=image[y:y+h,x:x+w]

#cv2.imshow("roi", roi)
#cv2.waitKey(0)
"""

