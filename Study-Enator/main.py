from CameraInterface.PiCamUtil import *

# import the necessary packages
import numpy as np
import cv2

path_to_file = 'C:/Users/Henok/source/repos/Study-Enator/Study-Enator/Accessories/'
prototxt = path_to_file + 'deploy.prototxt'
model = path_to_file + 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
weights = path_to_file + 'opencv_face_detector_uint8.pb'
image = path_to_file + 'elon.png'
min_confidence = .5

#load our serialized model from disk
net = cv2.dnn.readNetFromCaffe(prototxt, model)  

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it.

image = cv2.imread(image)
(h,w) =  image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
        (300,300),(104.0,177.0, 123.0))

# pass the blob thorugh the network and obtain the detections and predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

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

#show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
