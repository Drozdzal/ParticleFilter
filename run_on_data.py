import csv
import datetime
import math
import os

import cv2
import numpy as np
import scipy as scipy

from camera import Camera
from particle import Particle
from particle_filter import ParticleFilter
from shape import pitch_factory, Pitch
from transformations import rotation_x, translation
from input import prepare_observation2
from scipy.spatial.transform import Rotation as R


def get_rpy(axis_angle_rotation):
    rot = R.from_rotvec(np.array(axis_angle_rotation[:-1]) * axis_angle_rotation[-1]).as_euler('zxy')
    return rot


if __name__ == '__main__':
    pitch = Pitch()
    particle_count = 500
    np.random.seed(0)
    files = os.listdir('./data_30')
    files = [file for file in files if file.endswith('.jpg')]
    files.sort(key=lambda x: int(x[:-4]))
    # files = files[48:]
    actions = (['real_forward'] * 50 + ['real_right'] + ['real_forward'] * 25 + ['real_right']) * 40
    with open('./data_30/rotation_30.csv') as rotation_file, open('./data_30/translation_30.csv') as translation_file:
        rotation_reader = csv.reader(rotation_file)
        translation_reader = csv.reader(translation_file)
        rotations = [get_rpy([float(r) for r in row]) for row in rotation_reader]
        for rotation in rotations:
            rotation[2] = -rotation[2] + 3 * math.pi / 2
            while rotation[2] > 2 * math.pi:
                rotation[2] = rotation[2] - 2 * math.pi
        positions = [[float(r) for r in row] for row in translation_reader]
        positions = [[-1000 * pos[2], 1000 * pos[0]] for pos in positions]
        #
        # EXPERIMENT WITH NO FEEDBACK
        init_position = positions[1]
        init_rotation = rotations[1]
        init_particle = Particle(init_position,
                                 0,
                                 translation([0, 0, 500]).dot(rotation_x(- math.pi / 2 - math.pi / 6)),
                                 f=80)
        with open('output/no_feedback/general_results.csv', 'w', newline='') as general_csv, \
                open('output/no_feedback/particle_details.csv', 'w', newline='') as particles_csv:
            pf = ParticleFilter(init_particle, pitch, particle_count,
                                general_results_file=general_csv,
                                particle_details_file=particles_csv)
            pf.cluster_manager.add_dimension('x', min=-3000, max=3000, subdivs=30)
            pf.cluster_manager.add_dimension('y', min=-4500, max=4500, subdivs=45)
            pf.cluster_manager.add_dimension('yaw', min=0, max=2 * np.pi, subdivs=36, is_cyclic=True)
            for file in files[:500]:
                file_idx = int(file[:-4])
                print(file_idx)
                position = positions[file_idx + 1]
                rotation = rotations[file_idx + 1]
                action = actions[file_idx]
                # img = cv2.imread(f"./data_30/{file}")
                # cv2.imshow('raw_obs', cv2.resize(img, (200, 200)))
                # img = prepare_observation2(img)
                # pf.visualize()
                real_particle = Particle(position,
                                         rotation[2],
                                         translation([0, 0, 500]).dot(rotation_x(- math.pi / 2 - math.pi / 6)),
                                         f=80)
                pf.update_state(action, real_state=real_particle, render=False)
                pf.visualize(filename=f'./output/no_feedback/viz/{file_idx}.png', show=False)
                pf.save()
                # pf.update_observation(img)
                # pf.iterative_densest()


        # EXPERIMENT WITH FEEDBACK
        particle_count = 500
        init_position = positions[1]
        init_rotation = rotations[1]
        init_particle = Particle(init_position,
                                 0,
                                 translation([0, 0, 500]).dot(rotation_x(- math.pi / 2 - math.pi / 6)),
                                 f=80)
        with open('output/with_feedback/general_results.csv', 'w', newline='') as general_csv, \
                open('output/with_feedback/particle_details.csv', 'w', newline='') as particles_csv:
            pf = ParticleFilter(init_particle, pitch, particle_count,
                                general_results_file=general_csv,
                                particle_details_file=particles_csv)
            pf.cluster_manager.add_dimension('x', min=-3000, max=3000, subdivs=30)
            pf.cluster_manager.add_dimension('y', min=-4500, max=4500, subdivs=45)
            pf.cluster_manager.add_dimension('yaw', min=0, max=2 * np.pi, subdivs=36, is_cyclic=True)

            for file in files[:500]:
                file_idx = int(file[:-4])
                print(file_idx)
                position = positions[file_idx + 1]
                rotation = rotations[file_idx + 1]
                action = actions[file_idx]
                img = cv2.imread(f"./data_30/{file}")
                # cv2.imshow('raw_obs', cv2.resize(img, (200, 200)))
                img = prepare_observation2(img)
                # pf.visualize()
                real_particle = Particle(position,
                                         rotation[2],
                                         translation([0, 0, 500]).dot(rotation_x(- math.pi / 2 - math.pi / 6)),
                                         f=80)
                pf.update_state(action, real_state=real_particle)
                pf.visualize(filename=f'./output/with_feedback/viz/{file_idx}.png', show=False)
                pf.update_observation(img)
                pf.save()
                # pf.iterative_densest()
