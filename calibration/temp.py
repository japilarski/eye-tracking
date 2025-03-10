import numpy as np


def calculate_gaze_vector(pitch, yaw):
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    vector = np.array([0, 0, 1])
    pitch_matrix = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    yaw_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    intermediate_vector = np.dot(pitch_matrix, vector)
    final_vector = np.dot(yaw_matrix, intermediate_vector)
    return final_vector


angles = [
        (45, 45),  # Top-left corner
        (-45, -45),  # Bottom-right corner
        (-45, 45),  # Top-right corner
        (45, -45),  # Bottom-left corner
    ]

for pitch, yaw in angles:
    print(pitch, yaw)
    print(calculate_gaze_vector(pitch, yaw))
