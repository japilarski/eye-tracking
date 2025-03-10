import numpy as np
from screeninfo import get_monitors
import matplotlib.pyplot as plt


def check_monitor():
    monitor = get_monitors()[0]
    width = monitor.width
    height = monitor.height
    diagonal = np.sqrt(np.square(width) + np.square(height))
    display_sizes = {
        'width': width,
        'height': height,
        'diagonal': diagonal
    }
    return display_sizes


def temp(angles):
    columns = list(zip(*angles))

    # Find the smallest and largest values for each column
    min_values = [min(column) for column in columns]
    max_values = [max(column) for column in columns]
    return min_values, max_values


if __name__ == "__main__":
    # HARDCODED VALUES:
    '''
    Angles in spherical coordinate system, measured in angels [0, 360]
        1) pitch is measured between the z-axis and the radial line 
        2) yaw is measured between the orthogonal projection of the radial line r onto the reference x-y-plane
            and either of the fixed x-axis or y-axis
    '''
    angles = [
        (100, 50),  # Top-left corner
        (-100, -50),  # Bottom-right corner
        (100, -50),  # Top-right corner
        (-100, 50),  # Bottom-left corner
    ]
    display_size = check_monitor()
    # display_size = {'diagonal': np.sqrt(80)}

    test_angles = [
        (-50, 25)
    ]

    mi, mx = temp(angles)
    min_width, min_height = mi
    max_width, max_height = mx

    for pitch, yaw in test_angles:
        horizontal_position = ((pitch - min_width) / (max_width - min_width)) * display_size['width']
        vertical_position = ((yaw - min_height) / (max_height - min_height)) * display_size['height']
        print(horizontal_position, vertical_position)
        plt.plot(horizontal_position, vertical_position, 'x')

    screen_corners = [
        (0, 0),
        (0, display_size['height']),
        (display_size['width'], 0),
        (display_size['width'], display_size['height']),
    ]
    for y, z in screen_corners:
        plt.plot(float(y), float(z), 'o')

    plt.grid(True)
    plt.show()
