import sys

import pygame.event
import torch
from torch.utils.data import DataLoader, TensorDataset
from models import Autoencoder

from world import World
from render import Renderer

DEFAULT_GRID_WIDTH = 8
DEFAULT_GRID_HEIGHT = 8

BATCH_SIZE = 64


class LevelBase:

    def __init__(
        self,
        levelName: str = 'level',
    ):
        self.levelName = levelName

        self.width = DEFAULT_GRID_WIDTH
        self.height = DEFAULT_GRID_HEIGHT

        self.world = World(self.width, self.height)

        inputData = []
        for _ in range(1000):
            inputData.append(self.world.sample(reachable=False).tolist())
        print(f'inputdata[0]: {inputData[0]}')

        inputTensor = torch.tensor(inputData, dtype=torch.float32)
        print(f'inputTensor[0]: {inputTensor[0]}')

        tensorDataset = TensorDataset(inputTensor)

        tensorDataLoader = DataLoader(tensorDataset, batch_size=BATCH_SIZE, shuffle=True)

        autoEncoder = Autoencoder(input_size=4, latent_size=2, hidden=16)
        autoEncoder.train(tensorDataLoader, num_epochs=100)

        self.renderer = Renderer(world=self.world, scale=32, mmap_size=256, model=autoEncoder, marker=4)

        self.levelActive = True
        while self.levelActive:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.world.move_player(0, -1)
                    elif event.key == pygame.K_DOWN:
                        self.world.move_player(0, 1)
                    elif event.key == pygame.K_LEFT:
                        self.world.move_player(-1, 0)
                    elif event.key == pygame.K_RIGHT:
                        self.world.move_player(1, 0)
            self.world.step()
            self.renderer.step()

    def generateGameState(self, latentPoint, model):
        latentTensor = torch.Tensor(latentPoint).unsqueeze(0)
        outputs = model.decoder(latentTensor)
        return outputs.squeeze().detach().numpy().reshape((self.width, self.height))

    def unloadLevel(self):
        self.levelActive = False
