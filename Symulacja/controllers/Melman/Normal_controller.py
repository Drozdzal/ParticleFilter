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
move_finished=False # warunek, zeby wykonal ruch (skoro kontroler zostal wywolany, to jeszcze nie skonczyl ruchu)
sampling_time=0.01 # probka czasowa (musi byc zgodna z ruchami Melsona z Matlaba
max_steps = 10000


number_of_steps = 5

    
Finish = False

sim_time_start = robot.getTime() #pobranie czasu symulacji na poczatku

print("START")

while robot.step(timestep) != -1:

    if move_finished == False: #sprawdzenie ruchu, czy zostal zakonczony ruch
        
        sim_time = robot.getTime() #pobranie aktualnego czasu symulacji
        step = int((sim_time - sim_time_start) / sampling_time)  # zamiana czasu symulacji na numer kroku
        print("numer kroku: ", step)
        
        
        #Finish = action.move_forward(robot, step, number_of_steps)
        Finish = action.turn_right(robot, step, number_of_steps)
        #Finish = action.single_step(robot, step)
        #Finish = action.crouch(robot, step)

        if step >= max_steps or Finish == True: #warunek czy skonczyc petle (ruch)
            move_finished = True
            break
