import torch
import random
import itertools 
import numpy as np

class World:
    class Player:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class Switch:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.state = 0

    class Coin:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    class NPC:
        def __init__(self, path, period):
            self.x, self.y = path[0]
            self.path = path
            self.idx = 0
            self.period = period
            self.fcount = 0

        def step(self):
            if self.fcount < self.period:
                self.fcount += 1
            else:
                self.idx = (self.idx + 1) % len(self.path)
                self.x, self.y = self.path[self.idx]
                self.fcount = 0
    class Crate:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    def __init__(self, config):
        self.width = config['width']
        self.height = config['height']
        self.walls = np.zeros((self.width, self.height))

        for wall in config['walls']:
            self.walls[wall[0], wall[1]] = 1
        
        # Setup world content
        self.player = self.Player(0, 0)
        self.prev_vel = 0

        self.switches = [self.Switch(pt[0], pt[1]) for pt in config['switches']]
        self.coins = [self.Coin(pt[0], pt[1]) for pt in config['coins']]
        self.npcs = [self.NPC(path, 30) for path in config['npcs']]
        self.goal = self.sample()

        self.state_size = 1 + len(self.switches) + len(self.npcs)
        self.all_states = self.precompute_states()
        
    def move_player(self, dx, dy):
        if dy < 0 and self.player.y > 0 and self.walls[self.player.x][self.player.y - 1] == 0:
            self.player.y -= 1
        elif dy > 0 and self.player.y < self.height - 1 and self.walls[self.player.x][self.player.y + 1] == 0:
            self.player.y += 1
        elif dx < 0 and self.player.x > 0 and self.walls[self.player.x - 1][self.player.y] == 0:
            self.player.x -= 1
        elif dx > 0 and self.player.x < self.width - 1 and self.walls[self.player.x + 1][self.player.y] == 0:
            self.player.x += 1
        else:
            return
        self.prev_vel = 1

    def step(self):
        for switch in self.switches:
            if self.player.x == switch.x and self.player.y == switch.y and self.prev_vel > 0:
                switch.state = 0 if switch.state == 1 else 1
        self.prev_vel = 0

        self.coins = [coin for coin in self.coins if self.player.x != coin.x or self.player.y != coin.y]
        for npc in self.npcs:
            npc.step()

        if self.goal.int().tolist() == self.get_state().int().tolist():
            print('You won!!')

    def sample(self):
        state = []
        
        for _ in range(len(self.switches)):
            state.append(random.choice([0, 1]))
        state.append(random.randint(0, len(self.coins) - 1) if len(self.coins) > 0 else 0)
        for npc in self.npcs:
            state.append(random.randint(0, len(npc.path) - 1))
        return torch.tensor(state, dtype=torch.float32)
    
    def get_state(self):
        state = []
        for switch in self.switches:
            state.append(switch.state)
        state.append(len(self.coins))
        for npc in self.npcs:
            state.append(npc.idx)
        return torch.tensor(state, dtype=torch.float32)
    
    def precompute_states(self):
        states = []
        switch_states = list(itertools.product(list(range(2)), repeat=len(self.switches)))
        final_states = []
        for c in range(len(self.coins) + 1):
            for state in switch_states:
                final_states.append(list(state) + [c])
        
        for npc in self.npcs:
            temp = []
            for i in range(len(npc.path)):
                for state in final_states:
                    temp.append(state + [i])
            final_states = temp

        return final_states