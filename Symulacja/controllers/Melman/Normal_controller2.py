import cv2
import time
import csv
import math
import numpy as np

from controller import Robot, Camera, Accelerometer
from ImageProcessing import Vision
from Moves import Move
from camera import Camera


#inicjaliacja do webotsow
robot = Robot()  # tworzenie klasy Robot
timestep = int(robot.getBasicTimeStep())
Camera_Observation = Vision(robot.getDevice("camera"), timestep)  # dodanie kamery
move = Move(robot)
move_finished=False
sampling_time=0.01

crouch = 'crouch.csv'

single_step_short = 'forward/single_step_L30_T400_CoM198.csv'
single_step_long = 'forward/single_step_L60_T400_CoM198.csv'

#FORWARD
start_left_long = 'forward/start_left_L60_T400_CoM198.csv'

transfer_left_long = 'forward/transfer_left_L60_T400_CoM198.csv'
transfer_right_long = 'forward/transfer_right_L60_T400_CoM198.csv'

end_left_long = 'forward/end_left_L60_T500_CoM198.csv'
end_right_long = 'forward/end_right_L60_T400_CoM198.csv'

#TURN LEFT
start_turn_left_10 = 'turn_left/start_left_L10_T400_CoM198.csv'

transfer_turn_left_left_10 = 'turn_left/transfer_left_L10_T400_CoM198.csv'
transfer_turn_left_right_10 = 'turn_left/transfer_right_L10_T400_CoM198.csv'

end_turn_left_10 = 'turn_left/end_left_L10_T500_CoM198.csv'

#jaki rodzaj ruchu:
move_path = 'forward'

#Jeśli robot ma iść do przodu, trzeba dać  move_path = 'forward' i zdefiniować ilość kroków w number_of_steps. 


if move_path == 'forward':
    number_of_steps = 10 #number_of_steps musi być liczbą parzystą!

    leg_step = 1
    forward = True
    start_forward = True
    going_forward = False
    finish_forward = False
    leg = 'right'
    max_steps = 1000
    
elif move_path == 'turn_left':
    number_of_steps = 10 #number_of_steps musi być liczbą parzystą!

    leg_step = 1
    turn_left = True
    start_turn_left = True
    going_turn_left = False
    finish_turn_left = False
    leg = 'right'
    max_steps = 1000
    
else:
    max_steps = move.count_steps(move_path)-10

Finish = False

sim_time_start = robot.getTime()

print("START")

while robot.step(timestep) != -1:

    if move_finished == False:
        
        sim_time = robot.getTime()
        step = int((sim_time - sim_time_start) / sampling_time)  # zamiana czasu symulacji na numer kroku
        print("numer kroku: ", step)
        not_finish = False
        
        if move_path == crouch or move_path == single_step_short or move_path == single_step_long:
            move.perform_move(move_path, step)
        
        if move_path == 'forward':
            
            if finish_forward == True: # KONIEC RUCHU DO PRZODU
                move.perform_move(end_left_long, step)
                if step >= 200:
                    Finish = True
            
            if going_forward == True: # RUCH DO PRZODU
                if leg == 'right':
                    move.perform_move(transfer_right_long, step)
                if leg == 'left':
                    move.perform_move(transfer_left_long, step)
                
                if step >= 200:
                    if leg_step%2 == 1:
                        leg = 'left'
                    else :
                        leg = 'right'
                    leg_step += 1
                    sim_time_start = robot.getTime()
                
                if step >= 200 and leg_step == number_of_steps:
                    going_forward = False
                    finish_forward = True
                    sim_time_start = robot.getTime()
        
            if start_forward == True: # POCZĄTEK RUCHU DO PRZODU
                move.perform_move(start_left_long, step)
                if step >= 330:
                    start_forward = False
                    going_forward = True
                    sim_time_start = robot.getTime()
                    
        if move_path == 'turn_left':
            
            if finish_turn_left == True: # KONIEC RUCHU DO PRZODU
                move.perform_move(end_turn_left_10, step)
                if step >= 210:
                    Finish = True
            
            if going_turn_left == True: # RUCH DO PRZODU
                if leg == 'right':
                    move.perform_move(transfer_turn_left_right_10, step)
                if leg == 'left':
                    move.perform_move(transfer_turn_left_left_10, step)
                
                if step >= 210:
                    if leg_step%2 == 1:
                        leg = 'left'
                    else :
                        leg = 'right'
                    leg_step += 1
                    sim_time_start = robot.getTime()
                
                if step >= 210 and leg_step == number_of_steps:
                    going_turn_left = False
                    finish_turn_left = True
                    sim_time_start = robot.getTime()
        
            if start_turn_left == True: # POCZĄTEK RUCHU DO PRZODU
                move.perform_move(start_turn_left_10, step)
                if step >= 340:
                    start_turn_left = False
                    going_turn_left = True
                    sim_time_start = robot.getTime()
        
        if step >= max_steps or Finish == True:
            move_finished = True
            break


