import pygame
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
from models import Autoencoder
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

# Set up the autoencoder
input_size = 3
latent_size = 2

models = [Autoencoder(input_size, latent_size, HIDDEN) for _ in range(1)]

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(models[0].parameters(), lr=0.001)

def preprocess_state(state):
    return torch.Tensor(state.flatten()).unsqueeze(0)

def generate_latent_representation(game_state, model):
    inputs = preprocess_state(game_state)
    latent = model.encoder(inputs)
    return latent.squeeze().detach().numpy()

def generate_random_state():
    tensor = torch.rand(3)
    tensor[2] = random.choice([0, 1])
    return tensor

for model in models:
    # Training loop
    for epoch in range(EPOCHS):
        # Generate random game state
        game_state = generate_random_state()
        
        # Forward pass
        inputs = preprocess_state(game_state)
        outputs = model(inputs)
        
        # Compute the loss
        loss = criterion(outputs, inputs)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print the loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

# Generate latent representation for a game state

# Generate game state from latent representation
def generate_game_state(latent, model):
    latent_tensor = torch.Tensor(latent).unsqueeze(0)
    outputs = model.decoder(latent_tensor)
    game_state = outputs.squeeze().detach().numpy().reshape((GRID_HEIGHT, GRID_WIDTH))
    return game_state

bounds = []
for model in models:
    BOUND_L, BOUND_R, BOUND_U, BOUND_D = (0, 0, 0, 0)
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            for c in [0, 1]:
                state = torch.tensor([x, y, c], dtype=torch.float32)    
                z = generate_latent_representation(state, model)
                BOUND_L = min(BOUND_L, z[0])
                BOUND_R = max(BOUND_R, z[0])
                BOUND_U = min(BOUND_U, z[1])
                BOUND_D = max(BOUND_D, z[1])
    bounds.append((BOUND_L, BOUND_R, BOUND_U, BOUND_D))

load = True

if load:
    model = torch.load('test.pkl')
    BOUND_L, BOUND_R, BOUND_U, BOUND_D = (0, 0, 0, 0)
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            for c in [0, 1]:
                state = torch.tensor([x, y, c], dtype=torch.float32)    
                z = generate_latent_representation(state, model)
                BOUND_L = min(BOUND_L, z[0])
                BOUND_R = max(BOUND_R, z[0])
                BOUND_U = min(BOUND_U, z[1])
                BOUND_D = max(BOUND_D, z[1])
    bounds = (BOUND_L, BOUND_R, BOUND_U, BOUND_D)

    render = Renderer(world, GRID_SIZE, MMAP_SIZE, model, bounds, MARKER)
else:
    torch.save(models[0], 'test.pkl')
    render = Renderer(world, GRID_SIZE, MMAP_SIZE, models[0], bounds[0], MARKER)

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