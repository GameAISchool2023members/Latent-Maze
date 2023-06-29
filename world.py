import torch
import random
import itertools 

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
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        # Setup player
        self.player = self.Player(0, 0)
        self.switches = [self.Switch(6, 6), self.Switch(4, 6)]
        self.coins = [self.Coin(3, 4), self.Coin(3, 5)] 
        self.prev_vel = 0
        
        self.state_size = 3 + len(self.switches)
        self.switch_states = []
        self.all_states = self.precompute_states()
        # Setup goal
        self.goal = self.sample()

    def move_player(self, dx, dy):
        if dy < 0 and self.player.y > 0:
            self.player.y -= 1
        elif dy > 0 and self.player.y < self.height - 1:
            self.player.y += 1
        elif dx < 0 and self.player.x > 0:
            self.player.x -= 1
        elif dx > 0 and self.player.x < self.width - 1:
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

        if self.goal.int().tolist() == self.get_state().int().tolist():
            print('You won!!')

    def sample(self, reachable=True):
        state = [random.randint(0, self.width - 1), random.randint(0, self.height - 1)]
        if not reachable:
            state = [random.uniform(0, self.width), random.uniform(0, self.height)]
        for _ in range(len(self.switches)):
            state.append(random.choice([0, 1]))
        state.append(random.randint(0, len(self.coins) - 1))
        return torch.tensor(state, dtype=torch.float32)
    
    def get_state(self):
        state = [self.player.x, self.player.y]
        for switch in self.switches:
            state.append(switch.state)
        state.append(len(self.coins))
        return torch.tensor(state, dtype=torch.float32)
    
    def precompute_states(self):
        states = []
        switch_states = list(itertools.product(list(range(2)), repeat=len(self.switches)))
        for x in range(self.width):
            for y in range(self.height):
                for state in switch_states:
                    states.append([x, y] + list(state))
        final_states = []
        for c in range(len(self.coins) + 1):
            for state in states:
                final_states.append(state + [c])
                print(state + [c])
        return final_states