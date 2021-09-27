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
    camera.pose = translation([0, -20, 10]).dot(camera.pose)
    camera.inv_pose = np.linalg.inv(camera.pose)
    rendered = camera.render(pitch)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    rendered.draw(img)
    img = prepare_observation(rendered)
    # cv2.imshow('nic', img)
    # cv2.waitKey()
    # img = dilate(img, 3)
    img = blur(img, (9,9))
    img = img/img.max() * 255
    img = np.array(img, dtype=np.uint8)
    cv2.imshow('nic', img)
    cv2.waitKey()
    particle = Particle([0, -20], 0, translation([0, 0, 10]).dot(rotation_x(- math.pi / 2 - 0.5)))

    particle_count = 1000
    pf = ParticleFilter(particle, pitch, particle_count)
    pf.cluster_manager.add_dimension('x', min=-30, max=30, subdivs=30)
    pf.cluster_manager.add_dimension('y', min=-45, max=45, subdivs=45)
    pf.cluster_manager.add_dimension('yaw', min=0, max=2 * np.pi, subdivs=36, is_cyclic=True)
    X = (np.random.rand(particle_count) - 0.5) * 60
    Y = (np.random.rand(particle_count) - 0.5) * 90
    yaws = (np.random.rand(particle_count) - 0.5) * 2 * math.pi
    particles = [Particle([X[i], Y[i]], yaws[i], translation([0, 0, 10]).dot(rotation_x(- math.pi / 2 - 0.5))) for i in range(particle_count)]
    pf.particles = particles
    pf.particles[0] = Particle([0, -20], 0, translation([0, 0, 10]).dot(rotation_x(- math.pi / 2 - 0.5)))

    while (1):
        pf.visualize()
        pf.update_state('noop')
        pf.visualize()
        pf.update_observation(img)
    pass