import pygame
import numpy as np

# Constants
WIDTH, HEIGHT = 400, 400
GRID_SIZE = 10
DAMPING = 0.99
NUM_ITERATIONS = 10
ADD_WATER_RATE = 0.1  # Rate at which water is added when mouse button is held
REMOVE_WATER_RATE = 0.1  # Rate at which water is removed when mouse button is held

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Create grid data
grid_size_x = WIDTH // GRID_SIZE
grid_size_y = HEIGHT // GRID_SIZE
water_height = np.zeros((grid_size_x, grid_size_y), dtype=float)
water_velocity_x = np.zeros((grid_size_x, grid_size_y), dtype=float)
water_velocity_y = np.zeros((grid_size_x, grid_size_y), dtype=float)

# Interactive variables
mouse_down = False

def simulate_water():
    global water_height, water_velocity_x, water_velocity_y

    # Apply forces (e.g., gravity)
    water_velocity_y[1:grid_size_x-1, :] += 0.1  # Add some upward force

    # Update water velocity and height
    for _ in range(NUM_ITERATIONS):
        water_velocity_x[1:grid_size_x-1, 1:grid_size_y-1] += (
            (water_height[0:grid_size_x-2, 1:grid_size_y-1] - water_height[2:grid_size_x, 1:grid_size_y-1]) * 0.1)
        water_velocity_y[1:grid_size_x-1, 1:grid_size_y-1] += (
            (water_height[1:grid_size_x-1, 0:grid_size_y-2] - water_height[1:grid_size_x-1, 2:grid_size_y]) * 0.1)
        water_height = water_height * DAMPING + (water_velocity_x + water_velocity_y) * (1 - DAMPING)

def draw_water():
    for i in range(grid_size_x):
        for j in range(grid_size_y):
            color = (0, 0, int(max(0, int(min(255, water_height[i, j] * 10)))))
            pygame.draw.rect(screen, color, (i * GRID_SIZE, j * GRID_SIZE, GRID_SIZE, GRID_SIZE))

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button pressed (add water)
                mouse_down = True
            if event.button == 3:  # Right mouse button pressed (remove water)
                mouse_down = True
        if event.type == pygame.MOUSEBUTTONUP:
            mouse_down = False

    if mouse_down:
        x, y = pygame.mouse.get_pos()
        i, j = x // GRID_SIZE, y // GRID_SIZE
        if event.button == 1:  # Left mouse button held (add water)
            water_height[i, j] += ADD_WATER_RATE
        if event.button == 3:  # Right mouse button held (remove water)
            water_height[i, j] -= REMOVE_WATER_RATE

    simulate_water()
    screen.fill((0, 0, 0))
    draw_water()
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
