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
        self.switch = self.Switch(8, 8)
        self.prev_vel = 0
        self.cstate = 0

        # Setup goal
        self.goal_x = 0
        self.goal_y = 0

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
        if self.player.x == self.switch.x and self.player.y == self.switch.y and self.prev_vel > 0:
            self.switch.state = 0 if self.switch.state == 1 else 1
            self.cstate = self.switch.state
        self.prev_vel = 0