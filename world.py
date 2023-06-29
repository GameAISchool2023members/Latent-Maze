import torch
import random

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
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        # Setup player
        self.player = self.Player(0, 0)
        self.switches = [self.Switch(8, 8), self.Switch(4, 8)]
        self.prev_vel = 0
        
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

    def sample(self):
        tensor = torch.rand(2 + len(self.switches))
        for i in range(len(self.switches)):
            tensor[2 + i] = random.choice([0, 1])
        return tensor
    
    def get_state(self):
        state = [self.player.x, self.player.y]
        for switch in self.switches:
            state.append(switch.state)
        state = torch.tensor(state, dtype=torch.float32)
        return state