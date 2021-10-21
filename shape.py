import numpy as np
from line import Line
from typing import Dict, List
import cv2


class Shape:
    def __init__(self, lines: List[Line] = None):
        if lines is None:
            lines = []
        self.lines = lines
        self.polygons = []

    def draw(self, img, dir2color=False):
        for line in self.lines:
            line.draw(img, dir2color)
        width = img.shape[0]
        height = img.shape[1]
        for polygon in self.polygons:
            if len(polygon) < 2:
                continue
            points = []
            for line in polygon:
                points.append((int(line.a[0]) + width // 2, height // 2 + int(line.a[1])))
                points.append((int(line.b[0]) + width // 2, height // 2 + int(line.b[1])))
            points = np.array([points])
            # points = [(int(line.a[0]) + width // 2, height // 2 + int(line.a[1])) for line in polygon]
            cv2.fillPoly(img, points, (0, 0, 255))

    def draw_raw(self, img, dir2color=False):
        for line in self.lines:
            line.draw_raw(img, dir2color)

    def transform(self, transformation: np.array):
        for line in self.lines:
            line.a = transformation.dot(line.a)
            line.b = transformation.dot(line.b)

    def stroke(self, width: float):
        """
        creates a stroke from each line with given width in horizontal plane. Unnecessary and overengineered but cool.
        :param width:
        :return:
        """

        new_lines = []
        points_to_lines = {}
        for line in self.lines:
            if line.a[2] != 0 or line.b[2] != 0:
                new_lines.append(line)
                continue
            a = (line.a[0], line.a[1])
            b = (line.b[0], line.b[1])
            for point in [a, b]:
                if point in points_to_lines.keys():
                    points_to_lines[point].append(line)
                else:
                    points_to_lines[point] = [line]
        divs = []
        for line in self.lines:
            a = (line.a[0], line.a[1])
            b = (line.b[0], line.b[1])
            if line.a[2] != 0 or line.b[2] != 0:
                continue
            neighbouring_lines_a = [l for l in points_to_lines[a] if l != line]
            line_direction = (line.a - line.b)/line.get_length()
            line_normal = np.array([line_direction[1], -line_direction[0], 0, 0])
            dividers_a = []
            for neighbouring_line in neighbouring_lines_a:
                neighbour_direction = (neighbouring_line.b - neighbouring_line.a)/neighbouring_line.get_length()
                if (neighbouring_line.b == line.a).all():
                    neighbour_direction *= -1
                divider_normal = line_direction + neighbour_direction
                divider_normal /= np.linalg.norm(divider_normal)
                divider_direction = np.array([divider_normal[1], -divider_normal[0], 0, 0])
                divider_length = width/divider_direction.dot(line_normal)
                dividers_a.append(Line(line.a+divider_direction*divider_length, line.a - divider_direction*divider_length, (0, 0, 255)))

            neighbouring_lines_b = [l for l in points_to_lines[b] if l != line]
            line_direction = (line.b - line.a) / line.get_length()
            dividers_b = []
            for neighbouring_line in neighbouring_lines_b:
                neighbour_direction = (neighbouring_line.b - neighbouring_line.a) / neighbouring_line.get_length()
                if (neighbouring_line.b == line.b).all():
                    neighbour_direction *= -1
                divider_normal = line_direction + neighbour_direction
                divider_normal /= np.linalg.norm(divider_normal)
                divider_direction = np.array([divider_normal[1], -divider_normal[0], 0, 0])
                divider_length = width / divider_direction.dot(line_normal)
                dividers_b.append(
                    Line(line.b + divider_direction * divider_length, line.b - divider_direction * divider_length,
                         (0, 0, 255)))

            for divider in dividers_a:
                divs.append(divider)
            for divider in dividers_b:
                divs.append(divider)
            if not dividers_a or not dividers_b:
                continue
            left_border_a = max([divider.a if line_normal.dot(divider.a) > 0 else divider.b for divider in dividers_a],
                                key=lambda p: line_direction.dot(p))
            left_border_b = min([divider.a if line_normal.dot(divider.a) > 0 else divider.b for divider in dividers_b],
                                key=lambda p: line_direction.dot(p))
            new_lines.append(Line(left_border_a, left_border_b, (255, 255, 255)))

            right_border_a = max([divider.a if line_normal.dot(divider.a) < 0 else divider.b for divider in dividers_a],
                                key=lambda p: line_direction.dot(p))
            right_border_b = min([divider.a if line_normal.dot(divider.a) < 0 else divider.b for divider in dividers_b],
                                key=lambda p: line_direction.dot(p))
            new_lines.append(Line(right_border_a, right_border_b, (255, 255, 255)))
        self.lines = new_lines
        # for l in divs:
        #     self.lines.append(l)

    def subdivide(self):
        """
        divides a line into smaller lines if another line's end intersects it.
        """
        new_lines = []
        points = []
        points_list = []
        for line in self.lines:
            a = line.a.tolist()
            b = line.b.tolist()
            if a not in points_list:
                points.append(line.a)
                points_list.append(a)
            if b not in points_list:
                points.append(line.b)
                points_list.append(b)
        for line in self.lines:
            new_lines = new_lines + line.subdivide_multiple_points(points)
        self.lines = new_lines


class Pitch(Shape):
    def __init__(self, length=9000, width=6000, goal_width=1500, goal_height=750, penalty_width=2000, penalty_depth=500, radius=750, samples=16):
        super().__init__()

        color = (255, 255, 255)
        goal_width_outside = goal_width + 2 * 150
        goal_height_outside = goal_height + 150
        lines = [
            # borders
            Line([-width / 2, length / 2, 0, 1], [width / 2, length / 2, 0, 1], color),
            Line([width / 2, length / 2, 0, 1], [width / 2, -length / 2, 0, 1], color),
            Line([width / 2, -length / 2, 0, 1], [-width / 2, -length / 2, 0, 1], color),
            Line([-width / 2, -length / 2, 0, 1], [-width / 2, length / 2, 0, 1], color),
            # middle
            Line([-width / 2, 0, 0, 1], [width / 2, 0, 0, 1], color),

            # penalty areas
            Line([-penalty_width / 2, length / 2, 0, 1], [-penalty_width / 2, length / 2 - penalty_depth, 0, 1], color),
            Line([-penalty_width / 2, length / 2 - penalty_depth, 0, 1],
                 [penalty_width / 2, length / 2 - penalty_depth, 0, 1], color),
            Line([penalty_width / 2, length / 2, 0, 1], [penalty_width / 2, length / 2 - penalty_depth, 0, 1], color),

            Line([-penalty_width / 2, -length / 2, 0, 1], [-penalty_width / 2, -length / 2 + penalty_depth, 0, 1],
                 color),
            Line([-penalty_width / 2, -length / 2 + penalty_depth, 0, 1],
                 [penalty_width / 2, -length / 2 + penalty_depth, 0, 1], color),
            Line([penalty_width / 2, -length / 2, 0, 1], [penalty_width / 2, -length / 2 + penalty_depth, 0, 1], color)
        ]
        angles = list(np.linspace(0, 2 * np.pi, samples + 1))
        points = [[radius * np.cos(angle), radius * np.sin(angle), 0, 1] for angle in angles]
        points[-1] = points[0]
        for p1, p2 in zip(points[:-1], points[1:]):
            pass
            lines.append(Line(p1, p2, color))
        self.lines = lines
        self.subdivide()
        self.stroke(50)
        goals = [
            # GOALS
            # inside
            Line([-goal_width / 2, length / 2, goal_height, 1], [goal_width / 2, length / 2, goal_height, 1], color),
            Line([-goal_width / 2, -length / 2, goal_height, 1], [goal_width / 2, -length / 2, goal_height, 1], color),
            Line([-goal_width / 2, length / 2, 0, 1], [-goal_width / 2, length / 2, goal_height, 1], color),
            Line([goal_width / 2, length / 2, 0, 1], [goal_width / 2, length / 2, goal_height, 1], color),
            Line([-goal_width / 2, -length / 2, 0, 1], [-goal_width / 2, -length / 2, goal_height, 1], color),
            Line([goal_width / 2, -length / 2, 0, 1], [goal_width / 2, -length / 2, goal_height, 1], color),
            # outside
            Line([-goal_width_outside / 2, length / 2, goal_height_outside, 1],
                 [goal_width_outside / 2, length / 2, goal_height_outside, 1], color),
            Line([-goal_width_outside / 2, -length / 2, goal_height_outside, 1],
                 [goal_width_outside / 2, -length / 2, goal_height_outside, 1], color),
            Line([-goal_width_outside / 2, length / 2, 0, 1],
                 [-goal_width_outside / 2, length / 2, goal_height_outside, 1],
                 color),
            Line([goal_width_outside / 2, length / 2, 0, 1],
                 [goal_width_outside / 2, length / 2, goal_height_outside, 1],
                 color),
            Line([-goal_width_outside / 2, -length / 2, 0, 1],
                 [-goal_width_outside / 2, -length / 2, goal_height_outside, 1], color),
            Line([goal_width_outside / 2, -length / 2, 0, 1],
                 [goal_width_outside / 2, -length / 2, goal_height_outside, 1],
                 color),
        ]
        self.polygons.append([
            Line([-goal_width / 2, length / 2, goal_height, 1], [goal_width / 2, length / 2, goal_height, 1], color),
            # Line([goal_width / 2, length / 2, goal_height, 1], [goal_width / 2, length / 2, 0, 1], color),
            Line([goal_width / 2, length / 2, 0, 1], [-goal_width / 2, length / 2, 0, 1], color),
            # Line([-goal_width / 2, length / 2, 0, 1], [-goal_width / 2, length / 2, goal_height, 1], color),
        ])
        self.polygons.append([
            Line([-goal_width / 2, -length / 2, goal_height, 1], [goal_width / 2, -length / 2, goal_height, 1], color),
            # Line([goal_width / 2, -length / 2, goal_height, 1], [goal_width / 2, -length / 2, 0, 1], color),
            Line([goal_width / 2, -length / 2, 0, 1], [-goal_width / 2, -length / 2, 0, 1], color),
            # Line([-goal_width / 2, -length / 2, 0, 1], [-goal_width / 2, -length / 2, goal_height, 1], color),
        ])
        self.lines = self.lines + goals

def pitch_factory(length=9000, width=6000, goal_width=1500, goal_height=750, penalty_width=2000, penalty_depth=500, radius=750, samples=16) -> Shape:
    color = (255, 255, 255)
    goal_width_outside = goal_width + 2*150
    goal_height_outside = goal_height + 150
    lines = [
        # borders
        Line([-width / 2, length / 2, 0, 1], [width / 2, length / 2, 0, 1], color),
        Line([width / 2, length / 2, 0, 1], [width / 2, -length / 2, 0, 1], color),
        Line([width / 2, -length / 2, 0, 1], [-width / 2, -length / 2, 0, 1], color),
        Line([-width / 2, -length / 2, 0, 1], [-width / 2, length / 2, 0, 1], color),
        # middle
        Line([-width / 2, 0, 0, 1], [width / 2, 0, 0, 1], color),

        # penalty areas
        Line([-penalty_width / 2, length / 2, 0, 1], [-penalty_width / 2, length / 2 - penalty_depth, 0, 1], color),
        Line([-penalty_width / 2, length / 2 - penalty_depth, 0, 1], [penalty_width / 2, length / 2 - penalty_depth, 0, 1], color),
        Line([penalty_width / 2, length / 2, 0, 1], [penalty_width / 2, length / 2 - penalty_depth, 0, 1], color),

        Line([-penalty_width / 2, -length / 2, 0, 1], [-penalty_width / 2, -length / 2 + penalty_depth, 0, 1], color),
        Line([-penalty_width / 2, -length / 2 + penalty_depth, 0, 1],
             [penalty_width / 2, -length / 2 + penalty_depth, 0, 1], color),
        Line([penalty_width / 2, -length / 2, 0, 1], [penalty_width / 2, -length / 2 + penalty_depth, 0, 1], color)
    ]
    angles = list(np.linspace(0, 2 * np.pi, samples + 1))
    points = [[radius * np.cos(angle), radius*np.sin(angle), 0, 1] for angle in angles]
    points[-1] = points[0]
    for p1, p2 in zip(points[:-1], points[1:]):
        pass
        lines.append(Line(p1, p2, color))
    shape = Shape(lines)
    shape.subdivide()
    shape.stroke(50)

    goals = [
        # GOALS
        # inside
        Line([-goal_width / 2, length / 2, goal_height, 1], [goal_width / 2, length / 2, goal_height, 1], color),
        Line([-goal_width / 2, -length / 2, goal_height, 1], [goal_width / 2, -length / 2, goal_height, 1], color),
        Line([-goal_width / 2, length / 2, 0, 1], [-goal_width / 2, length / 2, goal_height, 1], color),
        Line([goal_width / 2, length / 2, 0, 1], [goal_width / 2, length / 2, goal_height, 1], color),
        Line([-goal_width / 2, -length / 2, 0, 1], [-goal_width / 2, -length / 2, goal_height, 1], color),
        Line([goal_width / 2, -length / 2, 0, 1], [goal_width / 2, -length / 2, goal_height, 1], color),
        # outside
        Line([-goal_width_outside / 2, length / 2, goal_height_outside, 1],
             [goal_width_outside / 2, length / 2, goal_height_outside, 1], color),
        Line([-goal_width_outside / 2, -length / 2, goal_height_outside, 1],
             [goal_width_outside / 2, -length / 2, goal_height_outside, 1], color),
        Line([-goal_width_outside / 2, length / 2, 0, 1], [-goal_width_outside / 2, length / 2, goal_height_outside, 1],
             color),
        Line([goal_width_outside / 2, length / 2, 0, 1], [goal_width_outside / 2, length / 2, goal_height_outside, 1],
             color),
        Line([-goal_width_outside / 2, -length / 2, 0, 1],
             [-goal_width_outside / 2, -length / 2, goal_height_outside, 1], color),
        Line([goal_width_outside / 2, -length / 2, 0, 1], [goal_width_outside / 2, -length / 2, goal_height_outside, 1],
             color),
    ]
    shape.lines = shape.lines + goals
    return shape

