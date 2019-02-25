import numpy as np
import cv2 as cv
#face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
face_cascade = cv.CascadeClassifier('C:/Users/khushal/PycharmProjects/BSE_analysis/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
eye_cascade = cv.CascadeClassifier('C:/Users/khushal/PycharmProjects/BSE_analysis/venv/Lib/site-packages/cv2/data/haarcascade_eye.xml')
smile_cascade = cv.CascadeClassifier('C:/Users/khushal/PycharmProjects/BSE_analysis/venv/Lib/site-packages/cv2/data/haarcascade_smile.xml')
eyeglasses_cascade = cv.CascadeClassifier('C:/Users/khushal/PycharmProjects/BSE_analysis/venv/Lib/site-packages/cv2/data/haarcascade_eye_tree_eyeglasses.xml')
Path_of_img = 'C:/Users/khushal/Pictures/Faces.jpg'
# Load an color image in coloured manner
img = cv.imread(Path_of_img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#cv.imshow('img',gray)
#cv.waitKey(0)
#cv.destroyAllWindows()

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

#print the number of faces found
print('Faces found: ', len(faces))

for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    #eyes = eye_cascade.detectMultiScale(roi_gray)
    #for (ex,ey,ew,eh) in eyes:
    #    cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    eye_glass = eyeglasses_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eye_glass:
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    smile = smile_cascade.detectMultiScale(roi_gray,scaleFactor=4.3, minNeighbors=6, minSize=(15, 15),flags=cv.CASCADE_SCALE_IMAGE)
    #print("Smiles found", len(smile))
    for (ex,ey,ew,eh) in smile:
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


#img = cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

save_img = 'C:/Users/khushal/Pictures/Faces_Detection.jpg'
cv.imwrite(save_img,img)

cv.imshow('Faces_Detection',img)
cv.waitKey(0)
cv.destroyAllWindows()

#import pyimage
import pytesseract

from pytesseract import image_to_data as itd
from pytesseract import image_to_string as its

#print(dir(pyimage))
print(dir(itd),'/n',dir(its))