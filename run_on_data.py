import datetime
import math
import os

import cv2
import numpy as np

from camera import Camera
from particle import Particle
from particle_filter import ParticleFilter
from shape import pitch_factory, Pitch
from transformations import rotation_x, translation
from input import prepare_observation2


if __name__ == '__main__':
    pitch = Pitch()
    particle = Particle([0, -2000], 0, translation([0, 0, 500]).dot(rotation_x(- math.pi / 2)), f=80)
    particle_count = 200
    pf = ParticleFilter(particle, pitch, particle_count)
    pf.cluster_manager.add_dimension('x', min=-3000, max=3000, subdivs=30)
    pf.cluster_manager.add_dimension('y', min=-4500, max=4500, subdivs=45)
    pf.cluster_manager.add_dimension('yaw', min=0, max=2 * np.pi, subdivs=36, is_cyclic=True)
    np.random.seed(0)
    X = (np.random.rand(particle_count) - 0.5) * 6000
    Y = (np.random.rand(particle_count) - 0.5) * 9000
    yaws = (np.random.rand(particle_count) - 0.5) * 2 * math.pi
    particles = [Particle([X[i], Y[i]], yaws[i], translation([0, 0, 500]).dot(rotation_x(- math.pi / 2 - math.pi/6)), f=80) for i in range(particle_count)]
    pf.particles = particles
    # pf.particles[0] = Particle([0, -20], 0, translation([0, 0, 500]).dot(rotation_x(- math.pi / 2 - 0.5)))
    # pf.particles = []
    iteration = 0
    print(pf.cluster_manager.get_cluster_count())
    files = os.listdir('./data_30')
    files = [file for file in files if file.endswith('.jpg')]
    files.sort(key=lambda x: int(x[:-4]))
    for file in files:
        img = cv2.imread(f"./data_30/{file}")
        img = prepare_observation2(img)
        cv2.imshow('obs', img)
        pf.visualize()
        pf.update_state('noop')
        pf.visualize()
        pf.update_observation(img)
        iteration += 1
        if iteration > 1:
            pf.iterative_densest()
    pass