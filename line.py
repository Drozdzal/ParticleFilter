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
        # cv2.circle(img, p1, 2, (0,0,255))
        # cv2.circle(img, p2, 5, (0, 0, 255))

    def get_length(self):
        return np.linalg.norm(self.b - self.a)

    def contains(self, point, precision=0.01):
        v1 = point - self.a
        v2 = self.b - point
        if np.linalg.norm(v1) < precision or np.linalg.norm(v2) < precision:
            return False
        return np.linalg.norm(v1) * np.linalg.norm(v2) - v1.dot(v2) < 0.01

    def insert_point(self, point):

        return [Line(self.a, point, self.color), Line(point, self.b, self.color)]

    def subdivide_multiple_points(self, points):
        subdivided_lines = []
        # searching for any point that belongs to the line
        for point in points:
            if self.contains(point):
                subdivided_lines = self.insert_point(point)
                break
        if subdivided_lines:
            result = []
            for sub_line in subdivided_lines:
                result = result + sub_line.subdivide_multiple_points(points)
            return result
        else:
            return [self]


    def __str__(self):
        return f"a:\t{self.a}\nb:\t{self.b}"
