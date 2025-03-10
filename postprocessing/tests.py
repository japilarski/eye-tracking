#Code to check if detection is working correctly
import pygame

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

def draw_circles(screen, points, color):
    for point in points:
        pygame.draw.circle(screen, color, point, 5)

def simulation():

    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    BLACK = (0, 0, 0)

    pygame.init()

    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Eye Tracking Simulation')

    clock = pygame.time.Clock()
    running = True
    eye_screen_position = []

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Simulation Eye tracking on screen(mouse position)
        eye_screen_x, eye_screen_y = pygame.mouse.get_pos()
        eye_screen_position.append((eye_screen_x, eye_screen_y))

        screen.fill(WHITE)
        fixations, saccades = detect_fixations_saccades(eye_screen_position)
        draw_circles(screen, fixations, GREEN)
        draw_circles(screen, saccades, RED)
        pygame.draw.circle(screen, BLACK, (eye_screen_x, eye_screen_y), 5)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == '__main__':
    simulation()
