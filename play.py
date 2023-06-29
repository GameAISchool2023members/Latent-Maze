import pygame
import sys
import torch

from world import World
from render import Renderer

# World settings
GRID_WIDTH = 10
GRID_HEIGHT = 10

# Visual settings
GRID_SIZE = 32
MMAP_SIZE = 128
MARKER = 4

# Model settings
EPOCHS = 1000
HIDDEN = 8

world = World(GRID_WIDTH, GRID_HEIGHT)

model = torch.load('level2.pkl')
render = Renderer(world, GRID_SIZE, MMAP_SIZE, model, MARKER)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                world.move_player(0, -1)
            elif event.key == pygame.K_DOWN:
                world.move_player(0, 1)
            elif event.key == pygame.K_LEFT:
                world.move_player(-1, 0)
            elif event.key == pygame.K_RIGHT:
                world.move_player(1, 0)
            elif event.key == pygame.K_SPACE:
                world.switch.state = 0 if world.switch.state == 1 else 1
    
    world.step()
    render.step()