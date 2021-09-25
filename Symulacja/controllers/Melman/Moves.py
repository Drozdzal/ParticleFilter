import csv
import time

class Move:
    def __init__(self,robot):
        self.MotorNames = ["Right-Trunk-z", "Right-Trunk-x [hip]", "Right-Trunk-y", "Right-Shank-y", "Right-Foot-y",
                      "Right-Foot-x", "Left-Trunk-z", "Left-Trunk-x [hip]", "Left-Trunk-y", "Left-Shank-y",
                      "Left-Foot-y", "Left-Foot-x", "Right-Arm-y [shoulder]",
                      "Right-Arm-x", "Right-Forearm-y", "Left-Arm-y [shoulder]", "Left-Arm-x", "Left-Forearm-y",
                      "Head-z", "Head-y"]
        self.Motor_num = len(self.MotorNames) - 2  # narazie ta glowe odrzucmy
        self.motor = []
        for i in range(len(self.MotorNames)):
            self.motor.append(robot.getDevice(self.MotorNames[i]))
            
    def initialization(self):
        with open('test.csv', 'r') as file:
            r = csv.reader(file)
            for i in range(2):
                next(r)
            row = next(r)
            for i in range(self.Motor_num):
                self.motor[i].setPosition(float(row[i+1]))

    def perform_move(self, csv_path, current_step):
        with open(csv_path, 'r') as file:
            r = csv.reader(file)
            for i in range(current_step):
                next(r)
            row = next(r)
            for i in range(self.Motor_num):
                self.motor[i].setPosition(float(row[i + 1]))
    def head(self, z, y):
                self.motor[18].setPosition(z)
                self.motor[19].setPosition(y)
