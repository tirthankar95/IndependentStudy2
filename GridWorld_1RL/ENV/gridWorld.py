from enum import Enum  
import numpy as np 
import time

class GridElemetns(Enum):
    EMPTY = 0 
    BRICK = 1 
    AGENT = 2
    GOAL = 3
    MX_ELEMENT = 4

class Action(Enum):
    UP = 0 
    DOWN = 1 
    LEFT = 2
    RIGHT = 3
    MX_ACTION = 4

class GridWorld:
    def __init__(self, n = 4) -> None:
        self.grid_size = n
        self.matrix = [[GridElemetns.EMPTY for _ in range(self.grid_size)] for __  in range(self.grid_size)]
        self.step_cnt = 0
        self.mx_step = n * 100
    def addElement(self, x: int, y: int, type: GridElemetns):
        self.matrix[x][y] = type 
        if type == GridElemetns.AGENT:
            self.ax, self.ay = x, y 
    def __str__(self) -> str:
        s = ""
        for _ in range(self.grid_size):
            for __ in range(self.grid_size):
                s = s + str(self.matrix[_][__].value) + " "
            s = s + "\n"
        s = s + "\n"
        return s 
    def step(self, action: Action):
        prevX, prevY = self.ax, self.ay 
        reward, terminated, truncated = 0, False, False 
        self.step_cnt += 1;
        if self.step_cnt >= self.mx_step: truncated = True
        if action == Action.UP:
            self.ax = min(self.ax + 1, self.grid_size - 1)
        elif action == Action.DOWN:
            self.ax = max(self.ax - 1, 0)
        elif action == Action.LEFT:
            self.ay = max(self.ay - 1, 0)
        elif action == Action.RIGHT:
            self.ay = min(self.ay + 1, self.grid_size - 1)
        # Decision after taking an action.
        if self.matrix[self.ax][self.ay] == GridElemetns.BRICK:
            self.ax, self.ay = prevX, prevY 
        elif self.matrix[self.ax][self.ay] == GridElemetns.GOAL:
            reward, terminated = 1, True 
        else:
            self.matrix[prevX][prevY] = GridElemetns.EMPTY
            self.matrix[self.ax][self.ay] = GridElemetns.AGENT
        if truncated: reward = -1
        return self.matrix, reward, terminated, truncated
        

class GridWorldBuilder:
    # Minimum size of grid should be 4.
    def __init__(self, n = 4) -> None:
        self.n = n 
    def reset(self, n = 4):
        self.n = max(4, n)
        self.g = GridWorld(n)
        self.g.addElement(0, 2, GridElemetns.AGENT)
        self.g.addElement(n-2, n-1, GridElemetns.GOAL)
        bricks = [[1, 1], [n-1, 0]]
        [self.g.addElement(b[0], b[1], GridElemetns.BRICK) for b in bricks]
        return self.g.matrix
    def step(self, action: Action):
        return self.g.step(action)
    def __str__(self) -> str:
        return str(self.g)

if __name__ == '__main__':
    gB = GridWorldBuilder()
    episodes = 1
    np.random.seed(int(time.time()))
    for _ in range(episodes):
        obs = gB.reset(n = 4)
        total_reward = 0
        print(str(gB), 0)
        while True:
            action = Action(np.random.randint(Action.MX_ACTION.value))
            obs, reward, terminated, truncated = gB.step(action)
            print(f'Action {action.value}\n')
            print(str(gB), reward)
            total_reward += 1
            if terminated or truncated:
                print(f'Episode[{_}] -> {1/total_reward}')
                break 