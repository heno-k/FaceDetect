import cv2
import numpy as np

#Mostly creating this function for notetaking purposes.... Need to learn if IP Camera is necessary.
#Also I don't fully understand what this software is/how secure it is...
#TODO: Investigate this https connection and understand how it is getting stream data.
def CameraConnect():
    capture = cv2.VideoCapture('http://raspberrypi:8080/?action=stream')
    return capture
