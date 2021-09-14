from shape import Shape
import cv2
import numpy as np


def dilate(img, width):
    kernel = np.ones((width, width), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


def prepare_observation(observation_shape: Shape, target_res=(200, 200)):
    img = np.zeros((target_res[0], target_res[1], 3), dtype=np.uint8)
    observation_shape.draw(img, dir2color=True)
    return img
