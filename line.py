import math

import numpy as np
import cv2


class Line:
    def __init__(self, a, b, color):
        self.a = np.array(a)
        self.b = np.array(b)
        self.color = color
        self.coordinates = None

    def get_coordinates(self):
        if self.coordinates is None:
            self.coordinates = np.array([self.a, self.b]).transpose()
        return self.coordinates

    def draw(self, img, dir2color=False):
        width = img.shape[0]
        height = img.shape[1]
        p1 = (int(self.a[0]) + width // 2, height // 2 + int(self.a[1]))
        p2 = (int(self.b[0]) + width // 2, height // 2 + int(self.b[1]))
        if dir2color:
            vector = self.b - self.a
            # direction = math.atan2(vector[0], vector[1])
            # color =
            vector_length = np.linalg.norm(vector)
            color = (int(abs(vector[0]/vector_length) * 255),
                     int(abs(vector[1]/vector_length) * 255),
                     0)
        else:
            color = self.color
        cv2.line(img, p1, p2, color)

