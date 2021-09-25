from controller import Robot,Camera,Accelerometer
from ImageProcessing import Vision
from Moves import Move
import cv2
import time

robot = Robot()                                        #tworzenie klasy Robot
timestep = int(robot.getBasicTimeStep())
camera=Vision(robot.getDevice("camera"), timestep)     #dodanie kamery
accelerometer = Accelerometer("accelerometer")
move=Move(robot)  

class Decision:
    def __init__(self,robot):
#        self.MotorNames = ["Right-Trunk-z", "Right-Trunk-x [hip]", "Right-Trunk-y", "Right-Shank-y", "Right-Foot-y",
#                      "Right-Foot-x", "Left-Trunk-z", "Left-Trunk-x [hip]", "Left-Trunk-y", "Left-Shank-y",
#                      "Left-Foot-y", "Left-Foot-x", "Right-Arm-y [shoulder]",
#                      "Right-Arm-x", "Right-Forearm-y", "Left-Arm-y [shoulder]", "Left-Arm-x", "Left-Forearm-y",
#                      "Head-z", "Head-y"]
#        self.Motor_num = len(self.MotorNames) - 2  # narazie ta glowe odrzucmy
#        self.motor = []
#        for i in range(len(self.MotorNames)):
#            self.motor.append(robot.getDevice(self.MotorNames[i]))
            
    def decision_making(self):
        if accelerometer.getValues(1)> 9:                 #robot leży na brzuchu
            move.StandUp_front
        if accelerometer.getValues(1)< -9:                #robot leży na plecach
            move.StandUp_back
            
        is_ball_found,circles=camera.GetBall()            #pobranie informacji czy znalazł piłkę (jeśli tak, to zbiera współrzędne okręgu gdzie została znaleziona
        
        if is_ball_found == 0:                            #nie znalazł piłki, więc ją szuka
            move.Look4Ball
            
        if is_ball_found == 1:                            #znalazł piłkę
            distance = camera.DistanceToBall()            #obliczenie odległości do piłki
            
            if distance > 1:                              #jeśli dystans jest wiekszy niz 1, to idzie do przodu
                move.StepForward()
            if distance < 1:
                move.KickBall()                           #jeśli dystans jest mniejszy niz 1, to kopie
