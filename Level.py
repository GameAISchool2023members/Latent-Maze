import json
import sys

import pygame.event
import torch
from torch.utils.data import DataLoader, TensorDataset
from models import Autoencoder

from world import World
from render import Renderer

BATCH_SIZE = 64


class Level:

    def __init__(
        self,
        levelFile: str = ''
    ):
        levelData = dict()
        with open(levelFile) as jsonFile:
            levelData = json.load(jsonFile)

        self.levelName = levelData.get('name', '')

        self.world = World(config=levelData)

        inputData = []
        for _ in range(1000):
            inputData.append(self.world.sample().tolist())
        print(f'inputdata[0]: {inputData[0]}')

        inputTensor = torch.tensor(inputData, dtype=torch.float32)
        print(f'inputTensor[0]: {inputTensor[0]}')

        tensorDataset = TensorDataset(inputTensor)

        tensorDataLoader = DataLoader(tensorDataset, batch_size=BATCH_SIZE, shuffle=True)

        autoEncoder = Autoencoder(input_size=self.world.state_size, latent_size=2, hidden=16)
        autoEncoder.train(tensorDataLoader, num_epochs=100)

        self.renderer = Renderer(world=self.world, scale=32, mmap_size=256, model=autoEncoder, marker=4)

        self.levelActive = True
        clock = pygame.time.Clock()
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
            isWinState = self.world.step()
            if isWinState:
                self.unloadLevel()
            self.renderer.step()
            clock.tick(30)

    def unloadLevel(self):
        self.levelActive = False
