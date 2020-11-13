import cv2
import numpy as np
from enum import Enum

class ImSrc(Enum):
    ImageFile   = 0
    VideoFile   = 1
    PiVideoStream = 2
    WebCamVideoStream = 3



#Mostly creating this function for notetaking purposes.... Need to learn if IP Camera is necessary.
#Also I don't fully understand what this software is/how secure it is...
#TODO: Investigate this https connection and understand how it is getting stream data.
def CameraConnect():
    capture = cv2.VideoCapture('http://raspberrypi:8080/?action=stream')
    return capture


# Place a square around the face in the images
def BoxFace(detections, min_confidence, image, h, w):
    #loop over the detections
    for i in range(0, detections.shape[2]):
        #extract the confidence (i.e., probability) aassociated with the 
        #prediction
        confidence = detections[0, 0, i, 2]

        #filter out weak detections by ensuring the 'confidence' is 
        #greater than the minimum confidence
        if confidence > min_confidence:
            #compute the (x,y)-coordinates of the bounding box for the 
            #object
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")

            #draw the bouding box of the face along with the associated
            #probability
            text = "{:.2}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX,startY), (endX, endY), 
                    (0,0, 255),2)
            cv2.putText(image, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)