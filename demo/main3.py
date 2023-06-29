import pygame
import sys
import random
# Initialize Pygame
pygame.init()

# Set up the display
GRID_SIZE = 32
GRID_WIDTH = 10
GRID_HEIGHT = 10
MMAP_SIZE = 128 
MARKER = 4
SCREEN_WIDTH = GRID_WIDTH * GRID_SIZE + MMAP_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * GRID_SIZE
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Latent Maze')

# Set up colors
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
GRAY = (64, 64, 64)
YELLOW = (255, 255, 0)

# Set up the initial position of the ball
ball_x = 0
ball_y = 0

import torch
import torch.nn as nn
import torch.optim as optim

# Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(Autoencoder, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, latent_size)
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, input_size),
            nn.Sigmoid()  # Sigmoid activation for output
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Set up the autoencoder
input_size = 3#GRID_WIDTH * GRID_HEIGHT
latent_size = 2
autoencoder = Autoencoder(input_size, latent_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Convert the game state to a tensor and flatten it
def preprocess_state(state):
    return torch.Tensor(state.flatten()).unsqueeze(0)

# Generate random game state for training (replace with actual game states)
def generate_random_state():
    tensor = torch.rand(3)
    tensor[2] = random.choice([0, 1])
    return tensor #(GRID_HEIGHT, GRID_WIDTH))

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Generate random game state
    game_state = generate_random_state()
    
    # Forward pass
    inputs = preprocess_state(game_state)
    outputs = autoencoder(inputs)
    
    # Compute the loss
    loss = criterion(outputs, inputs)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Generate latent representation for a game state
def generate_latent_representation(game_state):
    inputs = preprocess_state(game_state)
    latent = autoencoder.encoder(inputs)
    return latent.squeeze().detach().numpy()

# Generate game state from latent representation
def generate_game_state(latent):
    latent_tensor = torch.Tensor(latent).unsqueeze(0)
    outputs = autoencoder.decoder(latent_tensor)
    game_state = outputs.squeeze().detach().numpy().reshape((GRID_HEIGHT, GRID_WIDTH))
    return game_state

#print(generate_latent_representation(generate_random_state()))

GOAL_X, GOAL_Y = (6, 6)
MMAP_SCALE = 1 / 20

BOUND_L, BOUND_R, BOUND_U, BOUND_D = (0, 0, 0, 0)
for x in range(GRID_WIDTH):
    for y in range(GRID_HEIGHT):
        for c in [0, 1]:
            state = torch.tensor([x, y, c], dtype=torch.float32)    
            z = generate_latent_representation(state)
            BOUND_L = min(BOUND_L, z[0])
            BOUND_R = max(BOUND_R, z[0])
            BOUND_U = min(BOUND_U, z[1])
            BOUND_D = max(BOUND_D, z[1])
print(BOUND_L, BOUND_R, BOUND_U, BOUND_D)

def get_minipos(z):
    x = MMAP_SIZE * (z[0] - BOUND_L) / (BOUND_R - BOUND_L) 
    y = MMAP_SIZE * (z[1] - BOUND_U) / (BOUND_D - BOUND_U)
    return (x, y)

cstate = 0
# Main game loop
while True:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                if ball_y > 0:
                    ball_y -= 1
            elif event.key == pygame.K_DOWN:
                if ball_y < GRID_HEIGHT - 1:
                    ball_y += 1
            elif event.key == pygame.K_LEFT:
                if ball_x > 0:
                    ball_x -= 1
            elif event.key == pygame.K_RIGHT:
                if ball_x < GRID_WIDTH - 1:
                    ball_x += 1
            elif event.key == pygame.K_SPACE:
                cstate = 0 if cstate == 1 else 1

    # Draw the grid
    screen.fill(BLACK)
    
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            for c in [0, 1]:
                pygame.draw.rect(screen, (255, 255, 255), (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE), 1)
                state = torch.tensor([x, y, c], dtype=torch.float32)
                
                z = generate_latent_representation(state)
                mx, my = get_minipos(z)

                pygame.draw.circle(screen, GRAY, ( \
                    GRID_WIDTH * GRID_SIZE + mx - (MARKER // 2), \
                    my - (MARKER // 2)), \
                    MARKER)

    state = torch.tensor([GOAL_X, GOAL_Y, 0], dtype=torch.float32)
    z = generate_latent_representation(state)
    mx, my = get_minipos(z)

    pygame.draw.circle(screen, YELLOW, ( \
        GRID_WIDTH * GRID_SIZE + mx - (MARKER // 2), \
        my - (MARKER // 2)), \
        MARKER)

    # Minimap
    pygame.draw.rect(screen, (255, 255, 255), (GRID_WIDTH * GRID_SIZE, 0, 128, 128), 1)
    
    # Draw the ball
    pygame.draw.circle(screen, GREEN, (ball_x * GRID_SIZE + GRID_SIZE // 2, ball_y * GRID_SIZE + GRID_SIZE // 2), GRID_SIZE // 2)

    state = torch.tensor([ball_x, ball_y, cstate], dtype=torch.float32)
    z = generate_latent_representation(state)
    mx, my = get_minipos(z)

    pygame.draw.circle(screen, RED, ( \
        GRID_WIDTH * GRID_SIZE + mx - (MARKER // 2), \
        my - (MARKER // 2)), \
        MARKER)

    # Update the display
    pygame.display.flip()
    