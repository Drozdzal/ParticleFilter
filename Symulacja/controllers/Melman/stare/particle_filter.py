import math
from collections import OrderedDict

import cv2

from camera import Camera
from line import Line
from particle import Particle
from shape import Shape
import numpy as np
import itertools
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

class ClusterManager:
    class DimensionConfig:
        def __init__(self, min, max, subdivs, is_cyclic=False):
            self.min = min
            self.max = max
            self.subdivs = subdivs
            self.size = (max-min)/subdivs
            self.is_cyclic = is_cyclic

    def __init__(self):
        self.dimensions = OrderedDict()
        self.clusters = {}
        self.best_cluster_idx = None

    def clear(self):
        self.clusters = {}
        self.best_cluster_idx = None

    def add_dimension(self, name, min, max, subdivs, is_cyclic=False):
        self.dimensions[name] = ClusterManager.DimensionConfig(min, max, subdivs, is_cyclic=is_cyclic)

    def add_particle(self, particle: Particle):
        indexes = {}
        for dim_name, value in particle.get_state().items():
            indexes[dim_name] = value//self.dimensions[dim_name].size
        indexes = str(indexes)
        if indexes in self.clusters.keys():
            self.clusters[indexes].append(particle)
        else:
            self.clusters[indexes] = [particle]
        if self.best_cluster_idx is None:
            self.best_cluster_idx = indexes
        else:
            if len(self.clusters[indexes]) > len(self.clusters[self.best_cluster_idx]):
                self.best_cluster_idx = indexes

    def get_best_cluster(self):
        if self.best_cluster_idx is not None:
            return self.best_cluster_idx
        self.best_cluster_idx = max(self.clusters.keys(), key=lambda cluster_idx: len(self.clusters[cluster_idx]))
        return self.best_cluster_idx

    def get_cluster_count(self):
        count = 1
        for dim in self.dimensions.values():
            count *= dim.subdivs
        return count

    def get_best_estimate(self):
        best_cluster = self.clusters[self.get_best_cluster()]
        mean_state = {}
        for particle in best_cluster:
            for dim_name, value in particle.get_state().items():
                if dim_name not in mean_state:
                    mean_state[dim_name] = value/len(best_cluster)
                else:
                    mean_state[dim_name] += value/len(best_cluster)
        mean_state['particles in cluster'] = len(best_cluster)
        return mean_state

class ParticleFilter:
    def __init__(self,
                 starting_particle: Particle,
                 environment: Shape,
                 particle_count: int = 1000,
                 render_resolution=(400, 400)):
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
        dot_product = np.sum(np.multiply(self.theoretical_observations, observation), 3)
        probabilities = np.sum(np.sum(dot_product, 2), 1)
        # probabilities = np.sum(np.abs(probabilities ** 2), axis=-1)**(1/2)
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