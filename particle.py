import math

import numpy as np

from camera import Camera
from line import Line
from shape import Shape
from transformations import translation, rotation_z, rotation_y


class ActionParams:
    def __init__(self, translation_mean,
                 translation_deviation,
                 rotation_mean: float,
                 rotation_deviation: float):
        self.translation_mean = np.array(translation_mean)
        self.translation_deviation = np.array(translation_deviation)
        self.rotation_mean = rotation_mean
        self.rotation_deviation = rotation_deviation


ACTIONS = {
    # WZGLĘDEM ROBOTA KIERUNEK DO PRZODU TO OŚ Y, BOK TO X

    'forward': ActionParams(translation_mean=[0, 500],
                            translation_deviation=[200, 200],
                            rotation_mean=0,
                            rotation_deviation=0.05),
    'back': ActionParams(translation_mean=[0, -500],
                         translation_deviation=[200, 200],
                         rotation_mean=0,
                         rotation_deviation=0.5),
    'left': ActionParams(translation_mean=[0, 0],
                         translation_deviation=[100, 100],
                         rotation_mean=math.pi / 12,
                         rotation_deviation=0.5),
    'right': ActionParams(translation_mean=[0, 0],
                          translation_deviation=[1, 1],
                          rotation_mean=-math.pi / 12,
                          rotation_deviation=0.5),
    'noop': ActionParams(translation_mean=[0, 0],
                         translation_deviation=[100, 100],
                         rotation_mean=0,
                         rotation_deviation=0.1)

}


class Particle:
    def __init__(self, position, yaw, camera_transformation: np.array = None):
        self.camera = None
        self.position = np.array(position)
        self.yaw = yaw
        self.camera_transformation = camera_transformation
        if camera_transformation is not None:
            self.update_camera()
        self.probability = None
        self.window_min = None
        self.window_max = None
        self.copies = 0

    def update_camera(self, f=150):
        if self.camera is None:
            self.camera = Camera(f)
        self.camera.pose = self.get_robot_pose().dot(self.camera_transformation)
        self.camera.inv_pose = np.linalg.inv(self.camera.pose)

    def get_robot_pose(self):
        return translation(np.append(self.position, 0)).dot(rotation_z(self.yaw))

    def update(self, action: str):
        if action not in ACTIONS.keys():
            raise ValueError(f"Invalid action {action}")
        action = ACTIONS[action]

        # local translation matrix
        robot_translation = np.random.normal(action.translation_mean, action.translation_deviation)
        self.position = self.get_robot_pose().dot(np.append(robot_translation, [0, 1]))[0:2]
        # global rotation matrix
        self.yaw += np.random.normal(action.rotation_mean, action.rotation_deviation)
        self.yaw -= (self.yaw//(2*np.pi)) * (2*np.pi)
        self.update_camera()

    def copy(self):
        particle = Particle(self.position.copy(), self.yaw)
        particle.camera_transformation = self.camera_transformation
        if self.camera is not None:
            camera = Camera(self.camera.f)
            camera.pose = self.camera.pose.copy()
            camera.inv_pose = self.camera.inv_pose.copy()
            particle.camera = camera
        return particle

    def get_shape(self, color=(255, 0, 0)):
        start = self.camera.pose[0:4, 3]
        end = self.camera.get_robot_pose().dot(np.array([0, 500, 0, 1]))
        return Shape([Line(start, end, color)])

    def get_state(self):
        return {
            'x': self.position[0],
            'y': self.position[1],
            'yaw': self.yaw
        }