import csv
import math
from collections import OrderedDict
from typing import Optional

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
                 render_resolution=(200, 200),
                 general_results_file=None,
                 particle_details_file=None
                 ):
        self.environment = environment
        self.particle_count = particle_count
        self.particles = [starting_particle.copy() for i in range(particle_count)]
        self.theoretical_observations = np.zeros((particle_count, render_resolution[0], render_resolution[1], 3), dtype=np.uint8)
        self.best_particle = None
        self.cluster_manager = ClusterManager()
        self.real_state = None

        self.general_results_writer = None
        self.detailed_results_writer = None
        if general_results_file is not None:
            self.general_results_writer = csv.writer(general_results_file)
            self.general_results_writer.writerow(['real_x', 'real_y', 'real_yaw',
                                                  'cluster_x', 'cluster_y', 'cluster_yaw',
                                                  'densest_x', 'densest_y', 'densest_yaw',
                                                  'std_x', 'std_y', 'std_yaw'])
        if particle_details_file is not None:
            self.detailed_results_writer = csv.writer(particle_details_file)
            columns = []
            for i in range(particle_count):
                columns += [f'p{i}_x', f'p{i}_y', f'p{i}_yaw']
            self.detailed_results_writer.writerow(columns)

    def normalize_state(self, state):
        """
        Maps each state variable into 0-1 range
        :param state: Original state
        :return:
        """
        mapped = state - np.array([conf.min for conf in self.cluster_manager.dimensions.values()])
        return mapped / np.array([conf.max - conf.min for conf in self.cluster_manager.dimensions.values()])

    def iterative_densest(self, starting_point=None, iters=100):
        if starting_point is None:
            starting_point = starting_point or np.array([self.cluster_manager.get_best_estimate()[dim_name] for dim_name in self.cluster_manager.dimensions.keys()])
        points = np.array([list(particle.get_state().values()) for particle in self.particles])
        # normalization
        points = self.normalize_state(points)
        starting_point = self.normalize_state(starting_point)
        best_state = starting_point
        particle_distance = (1/self.cluster_manager.get_cluster_count()/len(self.cluster_manager.clusters[self.cluster_manager.best_cluster_idx]))**(1/3)
        std = particle_distance * 5
        for iteration in range(iters):
            differences = points - best_state
            differences[differences[:, 2] > 0.5, 2] = differences[differences[:, 2] > 0.5, 2] - 1
            differences[differences[:, 2] < -0.5, 2] = differences[differences[:, 2] < -0.5, 2] + 1
            distances = np.sum(differences ** 2, 1)
            gauss_values = np.exp(-distances / std / std)
            gradient = (differences.T * gauss_values).T
            best_state = best_state + np.mean(gradient, 0)
            pass
        # print(f'iterative best state: {best_state * np.array([conf.max - conf.min for conf in self.cluster_manager.dimensions.values()]) + np.array([conf.min for conf in self.cluster_manager.dimensions.values()])}')
        return best_state * np.array([conf.max - conf.min for conf in self.cluster_manager.dimensions.values()]) + \
                np.array([conf.min for conf in self.cluster_manager.dimensions.values()])

    def update_clusters(self):
        self.cluster_manager.clear()
        for particle in self.particles:
            self.cluster_manager.add_particle(particle)

    def update_state(self, action: str, real_state=None, render=True):
        self.real_state = real_state
        self.theoretical_observations *= 0
        self.cluster_manager.clear()
        for idx, particle in enumerate(self.particles):
            particle.update(action)
            self.cluster_manager.add_particle(particle)
            if render:
                particle.camera.render(self.environment).draw(self.theoretical_observations[idx, :, :], dir2color=True)
        # print(self.cluster_manager.get_best_estimate())

    def update_observation(self, observation: Optional[np.array] = None, alpha=1):
        cv2.imshow('current real observation', observation)
        observation = np.array(observation, dtype=np.uint64)
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
        cv2.imshow('matching regions', np.array(dot_product[best_idx], dtype=np.float)/dot_product[best_idx].max())

    def visualize(self, filename=None, show=True):
        camera = Camera(f=500)
        camera.pose = camera.pose.dot(translation([0, 0, 11000]))
        camera.pose = camera.pose.dot(rotation_x(math.pi))
        camera.inv_pose = np.linalg.inv(camera.pose)

        image = np.ones((500, 400, 3), dtype=np.uint8) * 255
        polys = self.environment.polygons
        self.environment.polygons = []
        camera.render(self.environment).draw(image, color=(128, 128, 128))
        self.environment.polygons = polys

        for particle in self.particles:
            particle_shape = camera.render(particle.get_shape())
            particle_shape.draw(image)
            particle_shape.lines[0].draw_a(image)
        if self.real_state is not None:
            particle_shape = camera.render(self.real_state.get_shape(color=(0, 0, 255)))
            particle_shape.draw(image)
            particle_shape.lines[0].draw_a(image, color=(0, 0, 255))
        best_estimate = self.iterative_densest()
        print(best_estimate)
        best_estimate_particle = Particle([best_estimate[0:2]], best_estimate[2],
                                          translation([0, 0, 500]).dot(rotation_x(- math.pi / 2 - math.pi / 6)),
                                          f=80)
        best_estimate_particle_shape = camera.render(best_estimate_particle.get_shape(color=(200, 0, 0)))
        best_estimate_particle_shape.draw(image)
        best_estimate_particle_shape.lines[0].draw_a(image, color=(200, 0, 0))
        if filename is not None:
            cv2.imwrite(filename, image)
        if show:
            cv2.imshow('visualization', image)
            cv2.waitKey(0)

    def save(self):
        real_pos = self.real_state.get_state()
        cluster_estimate = self.cluster_manager.get_best_estimate()
        densest_estimate = self.iterative_densest()
        std_pos_x = 0
        std_pos_y = 0
        std_yaw = 0
        if self.detailed_results_writer is not None:
            row = []
            for particle in self.particles:
                particle_state = particle.get_state()
                row = row + [particle_state['x'], particle_state['y'], particle_state['yaw']]
                std_pos_x += (particle_state['x'] - densest_estimate[0]) ** 2
                std_pos_y += (particle_state['y'] - densest_estimate[1]) ** 2
                std_yaw += (particle_state['yaw'] - densest_estimate[2]) ** 2
            self.detailed_results_writer.writerow(row)
        if self.general_results_writer is not None:

            self.general_results_writer.writerow([
                real_pos['x'], real_pos['y'], real_pos['yaw'],
                cluster_estimate['x'], cluster_estimate['y'], cluster_estimate['yaw'],
                densest_estimate[0], densest_estimate[1], densest_estimate[2],
                std_pos_x ** 0.5, std_pos_y ** 0.5, std_yaw ** 0.5
            ])
