import cv2
import time
import csv
import math
import numpy as np

#biblioteki do webots
from controller import Robot, Camera, Accelerometer
from ImageProcessing import Vision
from Moves import Move
from camera import Camera
from Actions import Action

# do filtru
from particle import Particle
from particle_filter import ParticleFilter
from shape import pitch_factory,Shape,Shape_Real
from transformations import rotation_x, translation

def dilate(img, width):
    kernel = np.ones((width, width), np.uint8)
    return cv2.dilate(img, kernel, iterations=3)

#inicjaliacja do webotsow
robot = Robot()  # tworzenie klasy Robot
timestep = int(robot.getBasicTimeStep())
Camera_Observation = Vision(robot.getDevice("camera"), timestep)  # dodanie kamery
move = Move(robot)
action = Action(robot,move)
number_of_steps = 2

#main Pawla
pitch = pitch_factory()
particle = Particle([0, -2000], 0, translation([0, 0, 500]).dot(rotation_x(- math.pi / 2 - 0.5)))
particle_count = 200
pf = ParticleFilter(particle, pitch, particle_count)
pf.cluster_manager.add_dimension('x', min=-3000, max=3000, subdivs=30)
pf.cluster_manager.add_dimension('y', min=-4500, max=4500, subdivs=45)
pf.cluster_manager.add_dimension('yaw', min=0, max=2 * np.pi, subdivs=36, is_cyclic=True)
np.random.seed(0)
X = (np.random.rand(particle_count) - 0.5) * 6000
Y = (np.random.rand(particle_count) - 0.5) * 9000
yaws = (np.random.rand(particle_count) - 0.5) * 2 * math.pi
particles = [Particle([X[i], Y[i]], yaws[i], translation([0, 0, 500]).dot(rotation_x(- math.pi / 2 - 0.5))) for i in
                 range(particle_count)]
pf.particles = particles
pf.particles[0] = Particle([0, -20], 0, translation([0, 0, 10]).dot(rotation_x(- math.pi / 2 - 0.5)))

iteration = 0
print(pf.cluster_manager.get_cluster_count())
while robot.step(timestep) != -1:

    lines = Camera_Observation.GetLines()
    Camera_img = Shape_Real(lines)

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
    Lines_from_camera = np.zeros((400, 400, 3), dtype=np.uint8)
    Camera_img.draw(Lines_from_camera, dir2color=True)

    pf.visualize()
    pf.update_state('noop')

    pf.visualize()
    pf.update_observation(Lines_from_camera)
    iteration += 1
    if iteration > 1:
        pf.iterative_densest()
    #czesc Pawla z filtrem





