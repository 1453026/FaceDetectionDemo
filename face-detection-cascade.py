import cv2
import sys

image_path = sys.argv[1]
casc_path = "haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(casc_path)

image = cv2.imread(image_path,cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)

print("Found {} faces".format(len(faces)))

for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("Face detection", image)
cv2.waitKey(0)
