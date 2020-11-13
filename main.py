from CameraInterface.PiCamUtil import *

# import the necessary packages
import numpy as np
import cv2

import imutils
import time
from imutils.video import VideoStream

Input_type = ImSrc.WebCamVideoStream

path_to_file = 'C:/Users/Henok/source/repos/FaceDetect/Accessories/'
prototxt_path = path_to_file + 'deploy.prototxt'
model_path = path_to_file + 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
weights_path = path_to_file + 'opencv_face_detector_uint8.pb'
min_confidence = .5

if(Input_type == ImSrc.ImageFile):
    image_path = path_to_file + 'elon.png'

#load our serialized model from disk
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)  

#Generate the blob depending on the input types (ImageFile, VidFile, VidStream)
if(Input_type == ImSrc.WebCamVideoStream or Input_type == ImSrc.PiVideoStream):
    #initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    if(Input_type == ImSrc.PiVideoStream):
        https_url = "http://raspberrypi:8080/?action=stream"
        vs = VideoStream(https_url).start()
        time.sleep(2.0)
    elif(Input_type == ImSrc.WebCamVideoStream):
        vs = VideoStream(src=0).start()
        time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        #grab the frame from the threaded video stream and resize it
        # to have a max width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # grab the frame dimensions and convert it to a blob
        (h,w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0,
                                     (300,300),(104.0, 177.0, 123.0))
        # pass the blob through the network and obtain the detections
        # and predictions
        net.setInput(blob)
        detections = net.forward()
        
        # place box around face
        BoxFace(detections, min_confidence, frame, h, w)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key was pressed, break from the loop
        if(key == ord("q")):
            break
    #do a bit of clean up
    cv2.destroyAllWindows()
    vs.stop()

elif(Input_type == ImSrc.ImageFile):
    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it.
    image = cv2.imread(image_path)
    (h,w) = image.shape[:2]
    blob  = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
            (300,300),(104.0,177.0, 123.0))

    # pass the blob thorugh the network and obtain the detections and predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    #place box around face
    BoxFace(detections, min_confidence, frame, h, w)

    #show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)
