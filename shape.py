import numpy as np
from line import Line
from typing import Dict, List
import cv2


class Shape:
    def __init__(self, lines: List[Line] = None):
        if lines is None:
            lines = []
        self.lines = lines

    def draw(self, img, dir2color=False):
        for line in self.lines:
            line.draw(img, dir2color)

    def transform(self, transformation: np.array):
        for line in self.lines:
            line.a = transformation * line.a
            line.b = transformation * line.b



def pitch_factory(length=90, width=60, goal_width=20, goal_height=10, radius=10, samples=16) -> Shape:
    color = (255, 255, 255)
    lines = [
        Line([-width / 2, length / 2, 0, 1], [width / 2, length / 2, 0, 1], color),
        Line([width / 2, length / 2, 0, 1], [width / 2, -length / 2, 0, 1], color),
        Line([width / 2, -length / 2, 0, 1], [-width / 2, -length / 2, 0, 1], color),
        Line([-width / 2, -length / 2, 0, 1], [-width / 2, length / 2, 0, 1], color),
        Line([-width / 2, 0, 0, 1], [width / 2, 0, 0, 1], color),
        Line([-goal_width / 2, length / 2, goal_height, 1], [goal_width / 2, length / 2, goal_height, 1], color),
        Line([-goal_width / 2, -length / 2, goal_height, 1], [goal_width / 2, -length / 2, goal_height, 1], color),
        Line([-goal_width / 2, length / 2, 0, 1], [-goal_width / 2, length / 2, goal_height, 1], color),
        Line([goal_width / 2, length / 2, 0, 1], [goal_width / 2, length / 2, goal_height, 1], color),
        Line([-goal_width / 2, -length / 2, 0, 1], [-goal_width / 2, -length / 2, goal_height, 1], color),
        Line([goal_width / 2, -length / 2, 0, 1], [goal_width / 2, -length / 2, goal_height, 1], color),
    ]
    angles = list(np.linspace(0, 2 * np.pi, samples + 1))
    points = [[radius * np.cos(angle), radius*np.sin(angle), 0, 1] for angle in angles]
    for p1, p2 in zip(points[:-1], points[1:]):
        pass
        lines.append(Line(p1, p2, color))
    return Shape(lines)

