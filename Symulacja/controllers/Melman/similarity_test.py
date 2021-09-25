import math

import cv2
import numpy as np

from camera import Camera
from particle import Particle
from particle_filter import ParticleFilter
from shape import pitch_factory
from transformations import rotation_x, translation

def show(name: str, img: np.array):
    converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow(name, converted)


def dilate(img, width):
    kernel = np.ones((width, width), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


def get_similarity(observation, observation_blurred, rendered):
    matching_area = cv2.bitwise_and(observation_blurred, rendered)
    matching_area = dilate(matching_area, 8)

    difference = cv2.bitwise_xor(observation, rendered)
    difference = cv2.bit
    cv2.imshow('diff', difference)
    cv2.waitKey()
    pass

if __name__ == '__main__':
    pitch = pitch_factory()
    camera = Camera()
    camera.pose = translation([0, -20, 10]).dot(rotation_x(- math.pi / 2 - 0.5))
    camera.inv_pose = np.linalg.inv(camera.pose)
    rendered = camera.render(pitch)

    camera.pose = translation([0, -25, 10]).dot(rotation_x(- math.pi / 2 - 0.4))
    camera.inv_pose = np.linalg.inv(camera.pose)
    rendered2 = camera.render(pitch)
    observation = np.zeros((200, 200), dtype=np.uint8)
    theoretical_observation = np.zeros((200, 200), dtype=np.uint8)
    rendered.draw(observation)
    rendered2.draw(theoretical_observation)
    observation_blurred = dilate(observation, 4)
    get_cost(observation, observation_blurred, theoretical_observation)

