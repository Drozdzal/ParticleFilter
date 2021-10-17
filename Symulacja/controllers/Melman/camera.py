import datetime
import math
from typing import Optional
import cv2
import numpy as np
from line import Line
from shape import Shape, pitch_factory
from transformations import translation, rotation_x, rotation_y, rotation_z


class Camera:
    def __init__(self, f=150):
        self.f = f
        self.pose = np.eye(4)
        self.inv_pose = np.eye(4)

    def render(self, shape: Shape, inv_pose: np.array = None) -> Shape:
        if inv_pose is None:
            inv_pose = self.inv_pose
        result = Shape()
        for line in shape.lines:
            new_line = self.project_line(line, inv_pose)
            if new_line is not None:
                result.lines.append(new_line)
        return result

    def project_line(self,
                     line: Line,
                     inv_pose: np.array = None,
                     cutoff_distance: float = 0.1) -> Optional[Line]:
        if inv_pose is None:
            inv_pose = self.inv_pose
        new_line_coords = inv_pose.dot(line.get_coordinates())
        coord_a = new_line_coords[:, 0]
        coord_b = new_line_coords[:, 1]
        if coord_a[2] < cutoff_distance and coord_b[2] < cutoff_distance:
            return None
        if coord_a[2] > cutoff_distance > coord_b[2]:
            coord_b = coord_a + (coord_b - coord_a) * (coord_a[2] - cutoff_distance)/(coord_a[2] - coord_b[2])
        if coord_b[2] > cutoff_distance > coord_a[2]:
            coord_a = coord_b + (coord_a - coord_b) * (coord_b[2] - cutoff_distance)/(coord_b[2] - coord_a[2])
        coord_a = coord_a / coord_a[2] * self.f
        coord_b = coord_b / coord_b[2] * self.f
        return Line(coord_a, coord_b, line.color)

    def get_robot_pose(self) -> np.array:
        y_vector = self.pose.dot(np.array([0, 1, 0, 0]))
        z_vector = self.pose.dot(np.array([0, 0, 1, 0]))
        angle = math.atan2(y_vector[2], z_vector[2])
        return self.pose.dot(rotation_x(-angle))
