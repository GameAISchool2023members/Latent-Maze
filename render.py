import pygame as pg
import torch

BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (64, 64, 64)
LGRAY = (128, 128, 128)

YELLOW = (255, 255, 0)

PAD_FACTOR = 0.2


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

        for state in self.world.all_states:
            state = torch.tensor(state, dtype=torch.float32)

            z = generate_latent_representation(state, self.model)
            BOUND_L = min(BOUND_L, z[0])
            BOUND_R = max(BOUND_R, z[0])
            BOUND_U = min(BOUND_U, z[1])
            BOUND_D = max(BOUND_D, z[1])

        BOUND_L -= abs(BOUND_L) * PAD_FACTOR
        BOUND_R += abs(BOUND_R) * PAD_FACTOR
        BOUND_U -= abs(BOUND_U) * PAD_FACTOR
        BOUND_D += abs(BOUND_D) * PAD_FACTOR

        return BOUND_L, BOUND_R, BOUND_U, BOUND_D

    def get_minipos(self, z):
        BOUND_L, BOUND_R, BOUND_U, BOUND_D = self.bounds
        x = self.mmap_size * (z[0] - BOUND_L) / (BOUND_R - BOUND_L)
        y = self.mmap_size * (z[1] - BOUND_U) / (BOUND_D - BOUND_U)
        return x, y

    def step(self):
        self.screen.fill(BLACK)

        for x in range(self.world.width):
            for y in range(self.world.height):
                if self.world.walls[x][y] == 1:
                    pg.draw.rect(self.screen, GRAY, (x * self.scale, y * self.scale, self.scale, self.scale))
                pg.draw.rect(self.screen, (255, 255, 255), (x * self.scale, y * self.scale, self.scale, self.scale), 1)

        switch_state = []

        for switch in self.world.switches:
            switch_state.append(switch.state)

        for state in self.world.all_states:
            state = torch.tensor(state, dtype=torch.float32)

            z = generate_latent_representation(state, self.model)
            mx, my = self.get_minipos(z)

            color = GRAY

            pg.draw.circle(
                self.screen,
                color,
                (self.world.width * self.scale + mx - (self.marker // 2), my - (self.marker // 2)),
                self.marker
            )

        z = generate_latent_representation(self.world.goal, self.model)
        mx, my = self.get_minipos(z)

        pg.draw.circle(
            self.screen,
            YELLOW,
            (self.world.width * self.scale + mx - (self.marker // 2), my - (self.marker // 2)),
            self.marker
        )

        for switch in self.world.switches:
            if switch.state == 0:
                pg.draw.line(
                    self.screen,
                    (255, 255, 255),
                    (switch.x * self.scale + self.scale / 2, switch.y * self.scale),
                    (switch.x * self.scale + self.scale / 2, (switch.y + 1) * self.scale)
                )
            else:
                pg.draw.line(
                    self.screen,
                    (255, 255, 255),
                    (switch.x * self.scale, switch.y * self.scale + self.scale / 2),
                    ((switch.x + 1) * self.scale, switch.y * self.scale + self.scale / 2)
                )

        for coin in self.world.coins:
            pg.draw.circle(
                self.screen,
                (200, 200, 0),
                (coin.x * self.scale + self.scale // 2, coin.y * self.scale + self.scale // 2),
                self.scale // 4
            )

        for npc in self.world.npcs:
            pg.draw.circle(
                self.screen,
                (100, 100, 255),
                (npc.path[npc.idx][0] * self.scale + self.scale // 2, npc.path[npc.idx][1] * self.scale + self.scale // 2),
                self.scale // 4
            )

        for crate in self.world.crates:
            pg.draw.rect(
                self.screen, 
                (0, 255, 0), 
                (crate.x * self.scale + 4, crate.y * self.scale + 4, self.scale - 8, self.scale - 8), 
                1
            )

        pg.draw.rect(
            self.screen,
            (255, 255, 255),
            (self.world.width * self.scale, 0, self.mmap_size, self.mmap_size),
            1
        )
        pg.draw.circle(
            self.screen,
            GREEN,
            (self.world.player.x * self.scale + self.scale // 2, self.world.player.y * self.scale + self.scale // 2),
            self.scale // 2
        )

        state = self.world.get_state()
        z = generate_latent_representation(state, self.model)
        mx, my = self.get_minipos(z)

        pg.draw.circle(
            self.screen,
            RED,
            (self.world.width * self.scale + mx - (self.marker // 2), my - (self.marker // 2)),
            self.marker
        )

        pg.display.flip()
