from controller import Robot, Camera, Accelerometer
from ImageProcessing import Vision
import cv2
import time
import csv
from Moves import Move
from camera import Camera
import math
from particle import Particle
from particle_filter import ParticleFilter
from shape import pitch_factory,Shape,Shape_Real
from transformations import rotation_x, translation
import numpy as np
def dilate(img, width):
    kernel = np.ones((width, width), np.uint8)
    return cv2.dilate(img, kernel, iterations=3)

#inicjaliacja do webotsow
robot = Robot()  # tworzenie klasy Robot
timestep = int(robot.getBasicTimeStep())
Camera_Observation = Vision(robot.getDevice("camera"), timestep)  # dodanie kamery
move = Move(robot)
move_finished=False
sampling_time=0.01

#main Pawla
pitch = pitch_factory()
camera = Camera()
particle = Particle([0, -20], 0, translation([0, 0, 10]).dot(rotation_x(- math.pi / 2 - 0.5)))
particle_count = 200
pf = ParticleFilter(particle, pitch, particle_count)
X = (np.random.rand(particle_count) - 0.5) * 60
Y = (np.random.rand(particle_count) - 0.5) * 90
yaws = (np.random.rand(particle_count) - 0.5) * 0.5 * math.pi
particles = [Particle([X[i], Y[i]], yaws[i], translation([0, 0, 10]).dot(rotation_x(- math.pi / 2 - 0.5))) for i in
                 range(particle_count)]
pf.particles = particles
pf.particles[0] = Particle([0, -20], 0, translation([0, 0, 10]).dot(rotation_x(- math.pi / 2 - 0.5)))
'''
Camera_img = Shape_Real(lines)
Lines_from_camera=np.zeros((640, 480, 3), dtype=np.uint8)
Camera_img.draw(Lines_from_camera,dir2color=True)'''

move_path = 'test.csv'
kroki = 590
sim_time_start = robot.getTime()
while robot.step(timestep) != -1:

    if move_finished == False:
        sim_time = robot.getTime()
        step = int((sim_time - sim_time_start) / sampling_time)  # zamiana czasu symulacji na numer kroku
        print(step)
        move.perform_move(move_path, step)
        if move_path == 'test.csv':
            if step >= kroki:
                move_finished = True


    if move_finished == True:
        lines = Camera_Observation.GetLines()
        Camera_img = Shape_Real(lines)
        Lines_from_camera = np.zeros((640, 480, 3), dtype=np.uint8)
        Camera_img.draw(Lines_from_camera, dir2color=True)
        '''
        #Jesli chcesz zobaczyc jak wykrywa linie odkomentuj
        img = Camera_Observation.CleanImg()
        s = Camera_Observation.MaskedLines()
        
        for line in lines:
            line=line[0]
            cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3)
        cv2.imshow("test",s)
        cv2.imshow("narysowane",img)
        cv2.waitKey()
        '''

        # obraz z symulacji i pogrubienie go

        pf.visualize()
        pf.update_state('noop')

        pf.visualize()
        pf.update_observation(Lines_from_camera)
    #czesc Pawla z filtrem





