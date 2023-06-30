import json
import sys

import pygame.event
import torch

from world import World
from render import Renderer

BATCH_SIZE = 64


class Level:

    def __init__(
        self,
        levelFile: str = ''
    ):
        self.levelData = dict()
        with open(f'{levelFile}.json') as jsonFile:
            self.levelData = json.load(jsonFile)

        levelModel = torch.load(f'{levelFile}.pkl')

        self.world = World(config=self.levelData)

        self.renderer = Renderer(world=self.world, scale=32, mmap_size=256, model=levelModel, marker=4)

        levelName = self.levelData.get('levelName', '')
        pygame.display.set_caption(f'Latent Maze — {levelName}')

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
                    elif event.key == pygame.K_r:
                        self.world = World(self.levelData)
                        self.renderer = Renderer(world=self.world, scale=32, mmap_size=256, model=levelModel, marker=4)
            isWinState = self.world.step()
            if isWinState:
                self.loadNextLevel()
            self.renderer.step()
            clock.tick(30)

    def loadNextLevel(self):
        pygame.display.set_caption('🎉 YOU WON! 🎉')
        self.levelActive = False
        nextLevel = self.levelData.get('nextLevel')
        if nextLevel:
            Level(nextLevel)
