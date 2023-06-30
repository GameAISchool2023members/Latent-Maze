import pygame
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
from models import Autoencoder, VAE
from world import World
from render import Renderer
from torch.utils.data import DataLoader, TensorDataset

# World settings
GRID_WIDTH = 8
GRID_HEIGHT = 8

# Visual settings
GRID_SIZE = 32
MMAP_SIZE = 256
MARKER = 4

# Model settings
EPOCHS = 1000
HIDDEN = 8

world = World(GRID_WIDTH, GRID_HEIGHT)

input_data = [world.sample().tolist() for _ in range(1000)]
input_tensor = torch.tensor(input_data, dtype=torch.float32)

dataset = TensorDataset(input_tensor)

batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#vae = VAE(3, 64, 2)
ae = Autoencoder(world.state_size, 2, 16)
ae.train(dataloader, 100)

#vae.train(dataloader, 10)

load = False

if load:
    model = torch.load('level2.pkl')
    render = Renderer(world, GRID_SIZE, MMAP_SIZE, model, MARKER)
else:
    #torch.save(models[0], 'test.pkl')
    render = Renderer(world, GRID_SIZE, MMAP_SIZE, ae, MARKER)

clock = pygame.time.Clock()

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
            elif event.key == pygame.K_r:
                world = World(GRID_WIDTH, GRID_HEIGHT)
                render = Renderer(world, GRID_SIZE, MMAP_SIZE, ae, MARKER)
    world.step()
    render.step()
    clock.tick(30)