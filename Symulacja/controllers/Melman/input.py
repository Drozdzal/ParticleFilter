from line import Line
from shape import Shape
import cv2
import numpy as np


def dilate(img, width):
    kernel = np.ones((width, width), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


def detect_circle(img: np.array, camera_transformation: np.array, f: int, circle_size):
    width, height = img.shape
    control_shape = Shape(lines=[
        Line([1, 1, 0, 1], [1, -1, 0, 1], (255, 255, 255)),
        Line([1, -1, 0, 1], [-1, -1, 0, 1], (255, 255, 255)),
        Line([-1, -1, 0, 1], [-1, 1, 0, 1], (255, 255, 255)),
        Line([-1, 1, 0, 1], [1, 1, 0, 1], (255, 255, 255)),
    ])
    # camera_direction =



def prepare_observation(observation_shape: Shape, target_res=(200, 200)):
    img = np.zeros((target_res[0], target_res[1], 3), dtype=np.uint8)
    observation_shape.draw(img, dir2color=True)
    return img
