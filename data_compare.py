import numpy as np

class Compare:
    def __init__(self):
        self.PITCH_LENGTH = 9000
        self.PITCH_WIDTH = 6000

    def average_particles_error(self, calculate_particles, real_particle):
        real_position = real_particle.position
        real_rotation = real_particle.yaw
        average_position = np.mean(calculate_particles.position)
        average_rotation = np.mean(calculate_particles.yaw)
        position_error = average_position - real_position
        rotate_error = average_rotation - real_rotation
        return position_error, rotate_error

    def best_particle_error(self, best_particle, real_particle):
        real_position = real_particle.position
        real_rotation = real_particle.yaw
        best_position = best_particle.position
        best_rotation = best_particle.yaw
        position_error = best_position - real_position
        rotate_error = best_rotation - real_rotation
        return position_error, rotate_error

if __name__ == '__main__':

    '''FIELD_LENGTH = 9000
    FIELD_WIDTH = 6000
    PI = np.pi
    error_absolute_average_position, error_absolute_average_rotation = average_particles_error(c_p,r_p)
    error_absolute_best_position, error_absolute_best_rotation = best_particle_error(b_p,r_p)

    error_relative_average_position = np.zeros(2, dtype=float)
    error_relative_best_position = np.zeros(2, dtype=float)

    error_relative_average_position[0] = error_absolute_average_position[0] / FIELD_LENGTH
    error_relative_average_position[1] = error_absolute_average_position[1] / FIELD_WIDTH
    error_relative_average_rotation = error_absolute_average_rotation / PI
    error_relative_best_position[0] = error_absolute_best_position[0] / FIELD_LENGTH
    error_relative_best_position[1] = error_absolute_best_position[1] / FIELD_WIDTH
    error_relative_best_rotation = error_absolute_best_rotation / PI'''
