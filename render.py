import pygame as pg
import torch

BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (64, 64, 64)
YELLOW = (255, 255, 0)

def preprocess_state(state):
    return torch.Tensor(state.flatten()).unsqueeze(0)

def generate_latent_representation(game_state, model):
    inputs = preprocess_state(game_state)
    latent = model.encoder(inputs)
    return latent.squeeze().detach().numpy()

class Renderer:
    def __init__(self, world, scale, mmap_size, model, marker):
        pg.init()
        self.world = world
        self.model = model
        self.bounds = self.find_bounds()
        
        SCREEN_WIDTH = world.width * scale + mmap_size
        SCREEN_HEIGHT = world.height * scale
        
        self.scale = scale
        self.mmap_size = mmap_size
        self.marker = marker

        self.screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pg.display.set_caption('Latent Maze')

    def find_bounds(self):
        BOUND_L, BOUND_R, BOUND_U, BOUND_D = (0, 0, 0, 0)
        for x in range(self.world.width):
            for y in range(self.world.height):
                for i in range(len(self.world.switches)):
                    for c in [0, 1]:
                        state = [x, y] + [0] * len(self.world.switches)
                        state[2 + i] = c
                        state = torch.tensor(state, dtype=torch.float32)
                        
                        z = generate_latent_representation(state, self.model)
                        BOUND_L = min(BOUND_L, z[0])
                        BOUND_R = max(BOUND_R, z[0])
                        BOUND_U = min(BOUND_U, z[1])
                        BOUND_D = max(BOUND_D, z[1])
        return (BOUND_L, BOUND_R, BOUND_U, BOUND_D)


    def get_minipos(self, z):
        BOUND_L, BOUND_R, BOUND_U, BOUND_D = self.bounds
        x = self.mmap_size * (z[0] - BOUND_L) / (BOUND_R - BOUND_L) 
        y = self.mmap_size * (z[1] - BOUND_U) / (BOUND_D - BOUND_U)
        return (x, y)
    
    def step(self):
        self.screen.fill(BLACK)
    
        for x in range(self.world.width):
            for y in range(self.world.height):
                for i in range(len(self.world.switches)):
                    for c in [0, 1]:
                        pg.draw.rect(self.screen, (255, 255, 255), (x * self.scale, y * self.scale, self.scale, self.scale), 1)
                        state = [x, y] + [0] * len(self.world.switches)
                        state[2 + i] = c
                        state = torch.tensor(state, dtype=torch.float32)
                        
                        z = generate_latent_representation(state, self.model)
                        mx, my = self.get_minipos(z)

                        pg.draw.circle(self.screen, GRAY, ( \
                            self.world.width * self.scale + mx - (self.marker // 2), \
                            my - (self.marker // 2)), self.marker)

        z = generate_latent_representation(self.world.goal, self.model)
        mx, my = self.get_minipos(z)

        pg.draw.circle(self.screen, YELLOW, ( \
            self.world.width * self.scale + mx - (self.marker // 2), \
            my - (self.marker // 2)), \
            self.marker)

        for switch in self.world.switches:
            if switch.state == 0:
                pg.draw.line(self.screen, (255, 255, 255), \
                            (switch.x * self.scale + self.scale / 2, switch.y * self.scale), \
                            (switch.x * self.scale + self.scale / 2, (switch.y + 1) * self.scale))
            else:
                pg.draw.line(self.screen, (255, 255, 255), \
                            (switch.x * self.scale, switch.y * self.scale + self.scale / 2), \
                            ((switch.x + 1) * self.scale, switch.y * self.scale + self.scale / 2))
            
        pg.draw.rect(self.screen, (255, 255, 255), (self.world.width * self.scale, 0, self.mmap_size, self.mmap_size), 1)        
        pg.draw.circle(self.screen, GREEN, (self.world.player.x * self.scale + self.scale // 2, self.world.player.y * self.scale + self.scale // 2), self.scale // 2)

        state = self.world.get_state()
        z = generate_latent_representation(state, self.model)
        mx, my = self.get_minipos(z)

        pg.draw.circle(self.screen, RED, ( \
            self.world.width * self.scale + mx - (self.marker // 2), \
            my - (self.marker // 2)), \
            self.marker)

        pg.display.flip()
        
