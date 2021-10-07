import cv2
import time
import csv
import math
import numpy as np

from controller import Robot, Camera, Accelerometer, Keyboard
from ImageProcessing import Vision
from Moves import Move
from camera import Camera
from Actions import Action


#inicjaliacja do webotsow
robot = Robot()  # tworzenie klasy Robot
timestep = int(robot.getBasicTimeStep())
Camera_Observation = Vision(robot.getDevice("camera"), timestep)  # dodanie kamery
move = Move(robot) #dodanie klasy Move
action = Action(robot,move) #dodanie klasy Action

keyboard = Keyboard()

number_of_steps = 1



print("START")


timestep = int(robot.getBasicTimeStep())

keyboard.enable(timestep)


while robot.step(timestep) != -1:

    key=keyboard.getKey()
        
    if key == 315:
        action.single_step(robot)
    elif key == 314:
        action.turn_left(robot, 1)
    elif key == 316:
        action.turn_right(robot, 1)
    elif key == 32:
        action.crouch(robot)


print("FINISH")

