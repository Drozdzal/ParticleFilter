from controller import Robot, Camera, Accelerometer
from ImageProcessing import Vision
from Moves import Move
import cv2
import time
import csv
from camera import Camera
from particle import Particle
from particle_filter import ParticleFilter
from shape import pitch_factory
from transformations import rotation_x, translation
from input import prepare_observation
import numpy as np

def empty(a):
    pass
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
cv2.createTrackbar("Hue Max","TrackBars",179,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",0,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",255,255,empty)
cv2.createTrackbar("Val Min","TrackBars",0,255,empty)
cv2.createTrackbar("Val Max","TrackBars",255,255,empty)
'''lower=np.array([112,44,108])
lower=np.array([111,13,110])
lower=np.array([108,67,106])
upper=np.array([179,255,255])
lower=np.array([0,0,74])
upper=np.array([179,255,255])
mask=cv2.inRange(imgHSV,lower,upper)
imgResult=cv2.bitwise_and(img,img,mask=mask)


lower=np.array([8,61,92])
upper=np.array([42,103,121])
'''



def dilate(img, width):
    kernel = np.ones((width, width), np.uint8)
    return cv2.dilate(img, kernel, iterations=3)

robot = Robot()  # tworzenie klasy Robot
timestep = int(robot.getBasicTimeStep())
Camera_Observation = Vision(robot.getDevice("camera"), timestep)  # dodanie kamery


while robot.step(timestep) != -1:
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    img = Camera_Observation.CleanImg()
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([h_min, s_min, v_min])
    upper_green = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower_green, upper_green)
    imgResult = cv2.bitwise_and(imgHSV, imgHSV, mask=mask)
    cv2.imshow('okienko',cv2.GaussianBlur(mask, (9, 9), 1))
    cv2.waitKey(1)
    '''
    img = Camera_Observation.MaskedGate()
    lines = Camera_Observation.GetLines()
    try:
        for line in lines:
            line = line[0]
            cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3)

    except:
        print("czegos nie widzi narazie")
    cv2.imshow("wsjo", img)'''
    cv2.waitKey(200)


accelerometer.disable()  # dezaktywacja akcelerometru

