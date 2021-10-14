import datetime
import math

import cv2
import numpy as np

from camera import Camera
from particle import Particle
from particle_filter import ParticleFilter
from shape import pitch_factory
from transformations import rotation_x, translation
from input import prepare_observation


def dilate(img, width):
    kernel = np.ones((width, width), np.uint8)
    return cv2.dilate(img, kernel, iterations=3)


def blur(img, size):
    return cv2.GaussianBlur(img, size, cv2.BORDER_DEFAULT)


if __name__ == '__main__':
    pitch = pitch_factory()
    camera = Camera()
    camera.pose = rotation_x(- math.pi / 2 - 0.5).dot(camera.pose)
    camera.pose = translation([0, -2000, 500]).dot(camera.pose)
    camera.inv_pose = np.linalg.inv(camera.pose)
    rendered = camera.render(pitch)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    rendered.draw(img)
    img = prepare_observation(rendered)
    # cv2.imshow('nic', img)
    # cv2.waitKey()
    # img = dilate(img, 3)
    img = blur(img, (15,15))
    img = img/img.max() * 255
    img = np.array(img, dtype=np.uint8)
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
    particles = [Particle([X[i], Y[i]], yaws[i], translation([0, 0, 500]).dot(rotation_x(- math.pi / 2 - 0.5))) for i in range(particle_count)]
    pf.particles = particles
    pf.particles[0] = Particle([0, -20], 0, translation([0, 0, 500]).dot(rotation_x(- math.pi / 2 - 0.5)))

    iteration = 0
    print(pf.cluster_manager.get_cluster_count())
    while (1):
        pf.visualize()
        pf.update_state('noop')
        pf.visualize()
        pf.update_observation(img)
        iteration += 1
        if iteration > 1:
            pf.iterative_densest()
    pass