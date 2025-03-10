import numpy as np
from screeninfo import get_monitors
import matplotlib.pyplot as plt


def calculate_gaze_vector(pitch, yaw):
    pitch_rad = np.deg2rad(pitch)
    yaw_rad = np.deg2rad(yaw)
    x = np.cos(yaw_rad) * np.sin(pitch_rad)
    y = np.sin(yaw_rad) * np.sin(pitch_rad)
    z = np.cos(pitch_rad)
    return np.array([x, y, z])


def calculate_angle_between_vectors(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle


def check_monitor():
    monitor = get_monitors()[0]
    width = monitor.width_mm
    height = monitor.height_mm
    diagonal = np.sqrt(np.square(width) + np.square(height))
    display_sizes = {
        'width': width,
        'height': height,
        'diagonal': diagonal
    }
    return display_sizes


def angles_to_position(distance, pitch, yaw):
    pitch_rad = np.deg2rad(pitch)
    yaw_rad = np.deg2rad(yaw)
    x = distance * np.cos(yaw_rad) * np.sin(pitch_rad)
    y = distance * np.sin(yaw_rad) * np.sin(pitch_rad)
    z = distance * np.cos(pitch_rad)
    return np.array([x, y, z])


if __name__ == "__main__":
    # HARDCODED VALUES:
    '''
    Angles in spherical coordinate system, measured in angels [0, 360]
        1) pitch is measured between the z-axis and the radial line 
        2) yaw is measured between the orthogonal projection of the radial line r onto the reference x-y-plane
            and either of the fixed x-axis or y-axis
    '''
    angles = [
        (79.48, 21.8),  # Top-left corner
        (180 - 79.48, -21.8),  # Bottom-right corner
        (79.48, -21.8),  # Top-right corner
        (180 - 79.48, 21.8),  # Bottom-left corner
    ]
    #         (55, 70),  # Top-center
    #         (70, 85),  # Right-center
    #         (60, 75),  # Bottom-center
    #         (65, 80),  # Left-center
    #         (30, 45)   # Center
    #     ]
    # display_size = check_monitor()
    display_size = {'diagonal': np.sqrt(80)}

    test_angles = [
        (0, 0)
    ]

    # for pitch, yaw in angles:
        # print(f'pitch: {pitch}, yaw: {yaw}')
        # print(calculate_gaze_vector(pitch, yaw))
    gaze_vectors = [calculate_gaze_vector(pitch, yaw) for pitch, yaw in angles]

    screen_distances = []
    for i in range(0, len(gaze_vectors)-1, 2):
        # print(i, i+1)
        v1 = gaze_vectors[i]
        v2 = gaze_vectors[i+1]
        theta = calculate_angle_between_vectors(v1, v2)
        distance = display_size['diagonal'] / (2 * np.tan(theta / 2))
        # print(distance)
        screen_distances.append(distance)

    average_distance = np.mean(screen_distances)
    print(f"Estimated distance from eye to screen center: {average_distance} units")

    real_values = [
        (4, 2),
        (-4, 2),
        (4, -2),
        (-4, -2),
    ]
    for y, z in real_values:
        plt.plot(float(y), float(z), 'o')

    for i, (pitch, yaw) in enumerate(angles):
        x, y, z = angles_to_position(average_distance, pitch, yaw)
        plt.plot(float(y), float(z), 'x', label=f'{i}')
        print('=' * 46)
        print('Angles:', pitch, yaw)
        print('Point: ', y, z)

    plt.grid(True)
    plt.show()
