import cv2
import time
import csv
import math
import numpy as np

from controller import Robot, Camera, Accelerometer
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


number_of_steps = 5



print("START")


action.move_forward(robot, number_of_steps)
action.turn_right(robot, 2)
action.turn_left(robot, 1)
action.single_step(robot)
action.crouch(robot)
action.single_step(robot)


print("FINISH")

