import csv
import datetime
import math
import os

import cv2
import numpy as np
import scipy as scipy

from camera import Camera
from line import Line
from particle import Particle, ACTIONS
from particle_filter import ParticleFilter
from shape import pitch_factory, Pitch, Shape
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
        # init_position = positions[1]
        # init_rotation = rotations[1]
        # init_particle = Particle(init_position,
        #                          0,
        #                          translation([0, 0, 500]).dot(rotation_x(- math.pi / 2 - math.pi / 6)),
        #                          f=80)
        # with open('output/no_feedback/general_results.csv', 'w', newline='') as general_csv, \
        #         open('output/no_feedback/particle_details.csv', 'w', newline='') as particles_csv:
        #     pf = ParticleFilter(init_particle, pitch, particle_count,
        #                         general_results_file=general_csv,
        #                         particle_details_file=particles_csv)
        #     pf.cluster_manager.add_dimension('x', min=-3000, max=3000, subdivs=30)
        #     pf.cluster_manager.add_dimension('y', min=-4500, max=4500, subdivs=45)
        #     pf.cluster_manager.add_dimension('yaw', min=0, max=2 * np.pi, subdivs=36, is_cyclic=True)
        #     for file in files[:500]:
        #         file_idx = int(file[:-4])
        #         print(file_idx)
        #         position = positions[file_idx + 1]
        #         rotation = rotations[file_idx + 1]
        #         action = actions[file_idx]
        #         img = cv2.imread(f"./data_30/{file}")
        #         # cv2.imshow('raw_obs', cv2.resize(img, (200, 200)))
        #         img = prepare_observation2(img)
        #         # pf.visualize()
        #         real_particle = Particle(position,
        #                                  rotation[2],
        #                                  translation([0, 0, 500]).dot(rotation_x(- math.pi / 2 - math.pi / 6)),
        #                                  f=80)
        #         pf.update_state(action, real_state=real_particle, render=False)
        #         pf.visualize(filename=f'./output/no_feedback/viz/{file_idx}.png', show=False)
        #         pf.save()
        #         # pf.update_observation(img)
        #         # pf.iterative_densest()


        # # EXPERIMENT WITH FEEDBACK
        # particle_count = 500
        # init_position = positions[1]
        # # init_position[1] -= 1000
        # init_rotation = rotations[1]
        # init_particle = Particle(init_position,
        #                          0,
        #                          translation([0, 0, 500]).dot(rotation_x(- math.pi / 2 - math.pi / 6)),
        #                          f=80)
        # with open('output/with_feedback/general_results.csv', 'w', newline='') as general_csv, \
        #         open('output/with_feedback/particle_details.csv', 'w', newline='') as particles_csv:
        #     pf = ParticleFilter(init_particle, pitch, particle_count,
        #                         general_results_file=general_csv,
        #                         particle_details_file=particles_csv)
        #     pf.cluster_manager.add_dimension('x', min=-3000, max=3000, subdivs=30)
        #     pf.cluster_manager.add_dimension('y', min=-4500, max=4500, subdivs=45)
        #     pf.cluster_manager.add_dimension('yaw', min=0, max=2 * np.pi, subdivs=36, is_cyclic=True)
        #
        #     for file in files[:500]:
        #         file_idx = int(file[:-4])
        #         print(file_idx)
        #         position = positions[file_idx + 1]
        #         rotation = rotations[file_idx + 1]
        #         action = actions[file_idx]
        #         img = cv2.imread(f"./data_30/{file}")
        #         # cv2.imshow('raw_obs', cv2.resize(img, (200, 200)))
        #         img = prepare_observation2(img)
        #         # pf.visualize()
        #         cv2.imshow('nic', img)
        #         cv2.waitKey(0)
        #         real_particle = Particle(position,
        #                                  rotation[2],
        #                                  translation([0, 0, 500]).dot(rotation_x(- math.pi / 2 - math.pi / 6)),
        #                                  f=80)
        #         pf.update_state(action, real_state=real_particle)
        #         pf.visualize(filename=f'./output/with_feedback/viz/{file_idx}.png', show=False)
        #         pf.update_observation(img)
        #         pf.save()
        #         # pf.iterative_densest()

        # SAVING REAL AND IDEAL ROBOT TRAJECTORY
        particle_count = 1
        for action in ACTIONS.values():
            action.rotation_deviation = action.rotation_deviation * 0
            action.translation_deviation = action.translation_deviation * 0
        init_position = positions[1]
        # init_position[1] -= 1000
        init_rotation = rotations[1]

        real_positions = [np.append(np.array(init_position), np.array([0, 1]))]
        ideal_positions = [np.append(np.array(init_position), np.array([0, 1]))]
        init_particle = Particle(init_position,
                                 0,
                                 translation([0, 0, 500]).dot(rotation_x(- math.pi / 2 - math.pi / 6)),
                                 f=80)
        pf = ParticleFilter(init_particle, pitch, particle_count)
        pf.cluster_manager.add_dimension('x', min=-3000, max=3000, subdivs=30)
        pf.cluster_manager.add_dimension('y', min=-4500, max=4500, subdivs=45)
        pf.cluster_manager.add_dimension('yaw', min=0, max=2 * np.pi, subdivs=36, is_cyclic=True)
        for file in files[:150]:
            file_idx = int(file[:-4])
            print(file_idx)
            position = positions[file_idx + 1]
            rotation = rotations[file_idx + 1]
            action = actions[file_idx]
            real_particle = Particle(position,
                                     rotation[2],
                                     translation([0, 0, 500]).dot(rotation_x(- math.pi / 2 - math.pi / 6)),
                                     f=80)
            pf.update_state(action, real_state=real_particle)
            real_positions.append(np.append(np.array(real_particle.position), np.array([0, 1])))
            ideal_positions.append(np.append(np.array(pf.particles[0].position), np.array([0, 1])))

        c = Camera(f=500)
        c.pose = c.pose.dot(translation([0, 0, 11000]))
        c.pose = c.pose.dot(rotation_x(math.pi))
        c.inv_pose = np.linalg.inv(c.pose)

        image = np.ones((500, 400, 3), dtype=np.uint8) * 255
        polys = pitch.polygons
        pf.environment.polygons = []
        c.render(pitch).draw(image, color=(128, 128, 128))
        pitch.polygons = polys
        real_pos_shape = Shape([])
        ideal_pos_shape = Shape([])

        for pos_idx in range(1, len(ideal_positions)):
            ideal_line = Line(ideal_positions[pos_idx-1], ideal_positions[pos_idx], (255, 0, 0))
            real_line = Line(real_positions[pos_idx - 1], real_positions[pos_idx], (0, 0, 255))
            ideal_pos_shape.lines.append(ideal_line)
            real_pos_shape.lines.append(real_line)
        c.render(real_pos_shape).draw(image)
        c.render(ideal_pos_shape).draw(image)
        #cv2.imshow('visualization', image)
        cv2.imwrite('output/real_ideal_comparison.png', image)
        #cv2.waitKey(0)
