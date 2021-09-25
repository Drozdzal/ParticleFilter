from controller import Robot,Camera,Accelerometer
from ImageProcessing import Vision
from Moves import Move
import cv2
import time
import csv

robot = Robot()                                          #tworzenie klasy Robot
timestep = int(robot.getBasicTimeStep())
camera=Vision(robot.getDevice("camera"), timestep)       #dodanie kamery
accelerometer = Accelerometer("accelerometer")           #dodanie akcelerometru
move=Move(robot)                                         #tworzenie obiektu ruchu

   
move.initialization()                                #pozycja wyjsciowa
time.sleep(2)

sampling_time = 0.010                                 #czas próbkowania w Matlabie

with open('test.csv') as f:                             #liczba kroków
    step_counter = sum(1 for line in f) - 1                                      
print(step_counter)

print("zaczynamy ruch")



accelerometer.enable(1)

while robot.step(timestep) != -1:

    step = 0
    
    sim_time_start = robot.getTime()                         #pobieranie czasu symulacji
    
    while robot.step(timestep) != -1:
        
        sim_time = robot.getTime()  
        step = int( (sim_time - sim_time_start) / sampling_time )
        
        if step>=step_counter:
            break
        
        if step == 0: step = 1
        move.crouch(step)                                #crouch - kucanie
        print("krok: ",step,", czas ruchu: ",round(sim_time - sim_time_start,3))
        print("odczyt z akcelerometru: ",accelerometer.getValues())
        camera.GetBall()
        
    #camera.GetBall()
            
    break
    
accelerometer.disable()

