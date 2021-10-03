import cv2
import time
import csv
import math
import numpy as np

from controller import Robot
from Moves import Move



class Action:
    def __init__(self,robot,move):
        self.leg_step = 1        
       
    def move_forward(self, robot, number_of_steps):
        sim_time_start = robot.getTime()
        timestep = int(robot.getBasicTimeStep())
        move = Move(robot)
        sampling_time=0.01
        Finish = False
        start_left_long = 'forward/start_left_L60_T400_CoM198.csv'
        transfer_left_long = 'forward/transfer_left_L60_T400_CoM198.csv'
        transfer_right_long = 'forward/transfer_right_L60_T400_CoM198.csv'
        end_left_long = 'forward/end_left_L60_T500_CoM198.csv'
        end_right_long = 'forward/end_right_L60_T400_CoM198.csv'
    
        leg_step = 0
        full_steps = 0
        start_forward = True
        going_forward = False
        finish_forward = False
        if number_of_steps%2 == 0: finish_left = True
        else: finish_left = False
        
        
        while robot.step(timestep) != -1:
        
            sim_time = robot.getTime() #pobranie aktualnego czasu symulacji
            step = int((sim_time - sim_time_start) / sampling_time)  # zamiana czasu symulacji na numer kroku
            print("numer kroku: ", step)
        
            if start_forward == True: # POCZĄTEK RUCHU DO PRZODU
                if step >= 337:
                    start_forward = False
                    going_forward = True

            if full_steps == number_of_steps+1:
                step = step-337    
                step = step%211
                going_forward = False
                finish_forward = True
                
            if step == 0: step = 1
            if finish_forward == True: # KONIEC RUCHU DO PRZODU
                if finish_left == True: move.perform_move(end_left_long, step)
                else: move.perform_move(end_right_long, step)
                if step >= 200:
                    Finish = True
                    break
                
            
            if going_forward == True: # RUCH DO PRZODU
                step = step-337
                leg_step = int(step/211)%2
                full_steps = int(step/211)
                step = step%211
                if step == 0: step = 1
                if leg_step == 0:
                    move.perform_move(transfer_right_long, step)
                if leg_step == 1:
                    move.perform_move(transfer_left_long, step)
                
            if start_forward == True: # POCZĄTEK RUCHU DO PRZODU
                move.perform_move(start_left_long, step)
            
        return Finish
        
        
        
    def turn_left(self, robot, number_of_steps):
    
        sim_time_start = robot.getTime()
        timestep = int(robot.getBasicTimeStep())
        move = Move(robot)
        sampling_time=0.01
        Finish = False
        start_left_long = 'turn_left/start_left_L10_T400_CoM198.csv'
        transfer_left_long = 'turn_left/transfer_left_L10_T400_CoM198.csv'
        transfer_right_long = 'turn_left/transfer_right_L10_T400_CoM198.csv'
        end_left_long = 'turn_left/end_left_L10_T500_CoM198.csv'
        end_right_long = 'turn_left/end_right_L10_T500_CoM198.csv'    
    
        leg_step = 0
        full_steps = 0
        start_turn_left = True
        going_turn_left = False
        finish_turn_left = False
        if number_of_steps%2 == 0: finish_left = True
        else: finish_left = False
        
        

        
        while robot.step(timestep) != -1:
        
            sim_time = robot.getTime() #pobranie aktualnego czasu symulacji
            step = int((sim_time - sim_time_start) / sampling_time)  # zamiana czasu symulacji na numer kroku
            print("numer kroku: ", step)
        
            if start_turn_left == True: # POCZĄTEK RUCHU DO PRZODU
                if step > 340:
                    start_turn_left = False
                    going_turn_left = True

                
            if full_steps == number_of_steps+1:
                step = step-337    
                step = step%211
                going_turn_left = False
                finish_turn_left = True
                
            if step == 0: step = 1
            if finish_turn_left == True: # KONIEC RUCHU DO PRZODU
                if finish_left == True: move.perform_move(end_left_long, step)
                else: move.perform_move(end_right_long, step)
                if step >= 200:
                    Finish = True
                    break
                
            
            if going_turn_left == True: # RUCH DO PRZODU
                step = step-340
                leg_step = int(step/211)%2
                full_steps = int(step/211)
                step = step%211
                if step == 0: step = 1
                if leg_step == 0:
                    move.perform_move(transfer_right_long, step)
                if leg_step == 1:
                    move.perform_move(transfer_left_long, step)
                
            if start_turn_left == True: # POCZĄTEK RUCHU DO PRZODU
                move.perform_move(start_left_long, step)
            
        return Finish
        
        
        
    def turn_right(self, robot, number_of_steps):
    
        sim_time_start = robot.getTime()
        timestep = int(robot.getBasicTimeStep())
        move = Move(robot)
        sampling_time=0.01
        Finish = False
        start_left_long = 'turn_right/start_left_L10_T400_CoM198.csv'
        transfer_left_long = 'turn_right/transfer_left_L10_T400_CoM198.csv'
        transfer_right_long = 'turn_right/transfer_right_L10_T400_CoM198.csv'
        end_left_long = 'turn_right/end_left_L10_T500_CoM198.csv'
        end_right_long = 'turn_right/end_right_L10_T500_CoM198.csv'
    
        leg_step = 0
        full_steps = 0
        start_turn_right = True
        going_turn_right = False
        finish_turn_right = False
        if number_of_steps%2 == 0: finish_left = True
        else: finish_left = False
        
        while robot.step(timestep) != -1:
        
        
            sim_time = robot.getTime() #pobranie aktualnego czasu symulacji
            step = int((sim_time - sim_time_start) / sampling_time)  # zamiana czasu symulacji na numer kroku
            print("numer kroku: ", step)
        
            if start_turn_right == True: # POCZĄTEK RUCHU DO PRZODU
                if step > 340:
                    start_turn_right = False
                    going_turn_right = True
                
            if full_steps == number_of_steps+1:
                step = step-340  
                step = step%215
                going_turn_right = False
                finish_turn_right = True
                
            if step == 0: step = 1
            if finish_turn_right == True: # KONIEC RUCHU DO PRZODU
                
                if finish_left == True: move.perform_move(end_left_long, step)
                else: move.perform_move(end_right_long, step)
                if step >= 200:
                    Finish = True
                    break
                
            
            if going_turn_right == True: # RUCH DO PRZODU
                step = step-340
                leg_step = int(step/215)%2
                full_steps = int(step/215)
                step = step%215
                if step == 0: step = 1
                
                if leg_step == 0:
                    move.perform_move(transfer_right_long, step)
                if leg_step == 1:
                    move.perform_move(transfer_left_long, step)
                
            if start_turn_right == True: # POCZĄTEK RUCHU DO PRZODU
                move.perform_move(start_left_long, step)
            
        return Finish
        
        
        
        

    def crouch(self, robot):
    
        sim_time_start = robot.getTime()
        timestep = int(robot.getBasicTimeStep())
        move = Move(robot)
        sampling_time=0.01
        Finish = False
        crouch = 'crouch.csv'
        
        while robot.step(timestep) != -1:
        
            sim_time = robot.getTime() #pobranie aktualnego czasu symulacji
            step = int((sim_time - sim_time_start) / sampling_time)  # zamiana czasu symulacji na numer kroku
            print("numer kroku: ", step)
        
            move.perform_move(crouch, step)
        
            if step >= 590:
                Finish = True
                break
        
        return Finish
        

        
    def single_step(self, robot):
    
        sim_time_start = robot.getTime()
        timestep = int(robot.getBasicTimeStep())
        move = Move(robot)
        sampling_time=0.01
        Finish = False
        single_step = 'forward/single_step_L60_T400_CoM198.csv'
        
        while robot.step(timestep) != -1:
        
            sim_time = robot.getTime() #pobranie aktualnego czasu symulacji
            step = int((sim_time - sim_time_start) / sampling_time)  # zamiana czasu symulacji na numer kroku
            print("numer kroku: ", step)
        
            move.perform_move(single_step, step)

        
            if step >= 400:
                Finish = True
                break
        
        return Finish
