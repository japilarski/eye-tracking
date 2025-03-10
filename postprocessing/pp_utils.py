# Functions from tests.py
def calculate_distance(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    return (dx**2 + dy**2)**0.5

def detect_fixations_saccades(eye_screen_position, threshold_saccade=5):
    fixations = []
    saccades = []

    for i in range(len(eye_screen_position) - 1):
        distance = calculate_distance(eye_screen_position[i], eye_screen_position[i + 1])

        if distance < threshold_saccade:
            fixations.append(eye_screen_position[i])
        else:
            saccades.append(eye_screen_position[i])

    return fixations, saccades