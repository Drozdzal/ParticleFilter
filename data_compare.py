import math

import numpy as np

class Compare:
    def __init__(self, f=100):
        self.FIELD_LENGTH = 640
        self.FIELD_WIDTH = 360
        self.inv_pose = np.eye(4)

    def average_particles_error(calculate_particles, real_particle):
        real_position = ...
        real_rotation = ...
        average_position = np.mean()
        average_rotation = np.mean()
        position_error = average_position - real_position
        rotate_error = average_rotation - real_rotation
        return position_error, rotate_error

    def best_particle_error(best_particle, real_particle):
        real_position = ...
        real_rotation = ...
        best_position = ...
        best_rotation = ...
        position_error = best_position - real_position
        rotate_error = best_rotation - real_rotation
        return position_error, rotate_error





if __name__ == '__main__':

    FIELD_LENGTH = 640
    FIELD_WIDTH = 360
    PI = np.pi
    #error_absolute_average_position, error_absolute_average_rotation = average_particles_error(c_p,r_p)
    #error_absolute_best_position, error_absolute_best_rotation = best_particle_error(b_p,r_p)

    error_relative_average_position = np.zeros(2, dtype=float)
    error_relative_best_position = np.zeros(2, dtype=float)

    '''error_relative_average_position[0] = error_absolute_average_position[0] / FIELD_LENGTH
    error_relative_average_position[1] = error_absolute_average_position[1] / FIELD_WIDTH
    error_relative_average_rotation = error_absolute_average_rotation / PI
    error_relative_best_position[0] = error_absolute_best_position[0] / FIELD_LENGTH
    error_relative_best_position[1] = error_absolute_best_position[1] / FIELD_WIDTH
    error_relative_best_rotation = error_absolute_best_rotation / PI'''
