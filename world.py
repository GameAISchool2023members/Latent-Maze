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
        self.player = self.Player(config['player_start'][0], config['player_start'][1])
        self.prev_vel = 0

        self.switches = [self.Switch(pt[0], pt[1]) for pt in config['switches']]
        self.coins = [self.Coin(pt[0], pt[1]) for pt in config['coins']]
        self.npcs = [self.NPC(path, 30) for path in config['npcs']]
        self.crates = [self.Crate(pt[0], pt[1]) for pt in config['crates']]
        self.goal = self.sample()

        self.state_size = 1 + len(self.switches) + len(self.npcs)
        self.all_states = self.precompute_states()
        
    def move_player(self, dx, dy):
        if self.resolve_crates(dx, dy):
            return
        
        solids = self.walls.copy()
        for crate in self.crates:
            solids[crate.x][crate.y] = 1

        if dy < 0 and self.player.y > 0 and solids[self.player.x][self.player.y - 1] == 0:
            self.player.y -= 1
        elif dy > 0 and self.player.y < self.height - 1 and solids[self.player.x][self.player.y + 1] == 0:
            self.player.y += 1
        elif dx < 0 and self.player.x > 0 and solids[self.player.x - 1][self.player.y] == 0:
            self.player.x -= 1
        elif dx > 0 and self.player.x < self.width - 1 and solids[self.player.x + 1][self.player.y] == 0:
            self.player.x += 1
        else:
            return
        self.prev_vel = 1

    def step(self) -> bool:
        for switch in self.switches:
            if self.player.x == switch.x and self.player.y == switch.y and self.prev_vel > 0:
                switch.state = 0 if switch.state == 1 else 1
        self.prev_vel = 0

        self.coins = [coin for coin in self.coins if self.player.x != coin.x or self.player.y != coin.y]
        for npc in self.npcs:
            npc.step()

        if self.goal.int().tolist() == self.get_state().int().tolist():
            print('You won!!')
            return True
        return False

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
    
    def resolve_crates(self, dx, dy):
        for target_crate in self.crates:
            if self.player.x + dx == target_crate.x and self.player.y + dy == target_crate.y:
                solids = self.walls.copy()
                for crate in self.crates:
                    solids[crate.x][crate.y] = 1
                for npc in self.npcs:
                    solids[npc.x][npc.y] = 1
                for switch in self.switches:
                    solids[switch.x][switch.y] = 1
                for coin in self.coins:
                    solids[coin.x][coin.y] = 1
                if target_crate.x + dx in range(self.width) and target_crate.y + dy in range(self.height) and solids[target_crate.x + dx][target_crate.y + dy] == 0:
                    target_crate.x += dx
                    target_crate.y += dy
                    return True 
        return False


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