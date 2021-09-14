import math

import cv2

from camera import Camera
from line import Line
from particle import Particle
from shape import Shape
import numpy as np

from transformations import translation, rotation_z, rotation_x


class ActionParams:
    def __init__(self, translation_mean,
                 translation_deviation,
                 rotation_mean: float,
                 rotation_deviation: float):
        self.translation_mean = np.array(translation_mean)
        self.translation_deviation = np.array(translation_deviation)
        self.rotation_mean = rotation_mean
        self.rotation_deviation = rotation_deviation


class ParticleFilter:
    def __init__(self,
                 starting_particle: Particle,
                 environment: Shape,
                 particle_count: int = 1000,
                 render_resolution=(200, 200)):
        self.environment = environment
        self.particle_count = particle_count
        self.particles = [starting_particle.copy() for i in range(particle_count)]
        self.theoretical_observations = np.zeros((particle_count, render_resolution[0], render_resolution[1], 3), dtype=np.uint8)
        self.best_particle = None

    def update_state(self, action: str):
        self.theoretical_observations *= 0
        for idx, particle in enumerate(self.particles):
            particle.update(action)
            particle.camera.render(self.environment).draw(self.theoretical_observations[idx, :, :], dir2color=True)

    def update_observation(self, observation: np.array, alpha=1):
        cv2.imshow('current real observation', observation)
        matching = np.multiply(self.theoretical_observations, observation)
        probabilities = np.sum(np.sum(matching * matching, 2), 1)
        probabilities = np.sum(np.abs(probabilities ** 2), axis=-1)**(1/2)
        probabilities = probabilities / sum(probabilities)
        window = 0
        pointer = 0
        resampled = []
        best_probability = 0
        best_idx = 0
        for idx, particle in enumerate(self.particles):
            particle.window_min = window
            if probabilities[idx] > best_probability:
                best_probability = probabilities[idx]
                best_idx = idx
                self.best_particle = particle
            particle.window_max = window + probabilities[idx]
            window += probabilities[idx]
            while particle.window_max > pointer:
                particle.copies += 1
                pointer += 1/self.particle_count
                resampled.append(particle.copy())
        self.particles = resampled[0:self.particle_count]
        cv2.imshow('best particle observation', self.theoretical_observations[best_idx])

    def visualize(self):
        camera = Camera(f=500)
        camera.pose = camera.pose.dot(translation([0, 0, 110]))
        camera.pose = camera.pose.dot(rotation_x(math.pi))
        camera.inv_pose = np.linalg.inv(camera.pose)

        image = np.zeros((500, 500, 3))
        camera.render(self.environment).draw(image)
        for particle in self.particles:
            camera.render(particle.get_shape()).draw(image)
        if self.best_particle is not None:
            camera.render(self.best_particle.get_shape(color=(0, 0, 255))).draw(image)
        cv2.imwrite('test.png', image)
        cv2.imshow('visualization', image)
        cv2.waitKey(0)