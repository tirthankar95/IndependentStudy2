#!/usr/bin/python3
import gym 
import sys 
sys.path.append('../')
import lbforaging 
import random
import time
import logging 
import numpy as np 
import copy
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s - Line %(lineno)d',
    level=logging.DEBUG
)

LOG = logging.getLogger(__name__)
from utils import * 
import pytest 
import random
import math 
from collections import defaultdict

render_env= False 
'''
ENVIRONMENT PARAMS.
'''
grid_size, num_agents, num_foods = 8, 2, 2
sight = 2 # Currently this is the only valid value.
episodes, episode_length = 1, 10
'''
        class Action(Enum):
            NONE = 0
            NORTH = 1
            SOUTH = 2
            WEST = 3
            EAST = 4
            LOAD = 5
'''
num_actions = 6
env = gym.make(f"Foraging-{sight}s-{grid_size}x{grid_size}-{num_agents}p-{num_foods}f-v2")


###########################################################################################

class State:
    def __init__(self, nobs, state = None, override = False) -> None:
        self.grid = np.zeros(shape = (grid_size, grid_size))
        self.mxfoodlvl = None 
        self.init_place_agents(nobs) # Place agents with appropriate levels.
        self.agent_food_check()
        self.init_place_foods(nobs) # Place food with appropriate levels; if possible.
        self.agent_food_check()
        if override:
            self.grid = state # for test purpose

    def getAgentLocation(self):
        agent = []
        agentLvL = []
        for _ in range(grid_size):
            for __ in range(grid_size):
                if self.grid[_][__] > 0:
                    lvl, idx = int(self.grid[_][__]//num_agents), \
                               int(self.grid[_][__]%num_agents)
                    agentLvL.append(lvl)
                    agent.append([idx, _, __, lvl])
        agent = sorted(agent, key = lambda x: x[0])
        agentLvL.sort()
        self.mxfoodlvl = sum(agentLvL[:min(3, num_agents)])
        return agent
    
    def init_place_foods(self, nobs):
        agent = self.getAgentLocation()
        correct = 0
        for aidx, obs in enumerate(nobs):
            for fidx in range(num_foods):
                iter_ = fidx * 3 
                if obs[iter_] == -1: continue 
                xf, yf, lvlf = obs[iter_: iter_+3]
                xa, ya = agent[aidx][1], agent[aidx][2]
                X, Y = int(xa + (xf - sight)), int(ya + (yf - sight))
                if X < 0 or Y < 0 or X >= grid_size\
                   or Y >= grid_size or int(self.grid[X][Y]) != 0\
                   or correct >= num_foods: continue
                self.grid[X][Y] = -lvlf 
                correct += 1
        while correct < num_foods:
            x, y = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
            while self.grid[x][y] != 0:
                x, y = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
            self.grid[x][y] = -random.randint(1, self.mxfoodlvl)
            correct += 1

    def init_place_agents(self, nobs):
        agent_lvl = []
        # Agent Level.
        for obs in nobs:
            offset = num_foods * 3 + 2
            agent_lvl.append(obs[offset])  
        # Food
        coord = {}; coord[(None, None)] = True 
        x, y = None, None 
        for aidx in range(num_agents):
            while (x, y) in coord:
                x, y = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
            coord[(x, y)] = True 
            self.grid[x][y] = agent_lvl[aidx] * num_agents + aidx
        
    def adjacent_food_location(self, row, col):
        if row > 0 and self.grid[row - 1][col] < 0:
            return row - 1, col
        elif row < grid_size - 1 and self.grid[row + 1][col] < 0:
            return row + 1, col
        elif col > 0 and self.grid[row][col - 1] < 0:
            return row, col - 1
        elif col < grid_size - 1 and self.grid[row][col + 1] < 0:
            return row, col + 1 
        return None, None
    
    def isValid(self, x, y):
        if x < 0 or y < 0 or x >= grid_size \
           or y >= grid_size or self.grid[x][y] < 0 \
           or self.grid[x][y] > 0: return False 
        return True 
    
    def agent_food_check(self):
        agent_c, food_c = 0, 0
        for _ in range(grid_size):
            for __ in range(grid_size):
                if self.grid[_][__] > 0: # It's an agent
                    agent_c += 1
                if self.grid[_][__] < 0:
                    food_c += 1
        assert agent_c == num_agents and food_c <= num_foods, \
               f'Agent[{agent_c}] Food[{food_c}]'
        
    def move(self, actions) -> None:
        agents = self.getAgentLocation()
        collisions = defaultdict(list)
        loading_agents = []
        for a_idx, agent in enumerate(agents):
            idx, x, y, level = agent 
            action = Action(actions[a_idx])
            if action == Action.NONE:
                collisions[(x, y)].append([x, y, level * num_agents + idx])
            elif action == Action.NORTH and self.isValid(x - 1, y):
                collisions[(x - 1, y)].append([x, y, level * num_agents + idx])
            elif action == Action.SOUTH and self.isValid(x + 1, y):
                collisions[(x + 1, y)].append([x, y, level * num_agents + idx])
            elif action == Action.WEST and self.isValid(x, y - 1):
                collisions[(x, y - 1)].append([x, y, level * num_agents + idx])
            elif action == Action.EAST and self.isValid(x, y + 1):
                collisions[(x, y + 1)].append([x, y, level * num_agents + idx])
            elif action == Action.LOAD:
                collisions[(x, y)].append([x, y, level * num_agents + idx])
                loading_agents.append([x, y, idx, level])

        self.agent_food_check()    
        for k, v in collisions.items():
            if len(v) > 1: continue 
            x, y, mix = v[0]
            self.grid[x][y] = 0
            self.grid[k[0]][k[1]] = mix

        self.agent_food_check()    
        n_LA = len(loading_agents)
        for idx, ip in enumerate(loading_agents):
            adj_players = [ agent for agent in agents 
                            if (abs(ip[0] - agent[0]) == 1
                            and ip[1] == agent[1])
                            or (abs(ip[1] - agent[1]) == 1
                            and ip[0] == agent[0]) 
                            and (idx != agent[2])]
            running_level = ip[-1]
            for adj_player in adj_players:
                if adj_player in loading_agents:
                    running_level += adj_player[-1]
            frow, fcol = self.adjacent_food_location(ip[0], ip[1])
            if frow == None or fcol == None: continue 
            flvl = -1 * self.grid[frow][fcol]
            if flvl <= running_level:
                self.grid[frow][fcol] = 0
                    
        
    def _transform_to_neighborhood(self, center, sight, position):
        return [
            position[0] - center[0] + sight,
            position[1] - center[1] + sight,
        ]
    @property
    def view(self):
        return self.grid 

    def get_neighbor_food(self, x, y):
        food = []
        foodCnt = 0
        offset_x, offset_y = max(0, x - sight), max(0, y - sight)
        for i in range(max(0, x - sight), min(grid_size, x + sight + 1)):
            for j in range(max(0, y - sight), min(grid_size, y + sight + 1)):
                if self.grid[i][j] < 0:
                    foodCnt += 1
                    food.extend([i-offset_x, j-offset_y, -1 * self.grid[i][j]])
        while foodCnt < num_foods:
            food.extend([-1, -1, 0])
            foodCnt += 1
        return food

    def nObs(self):
        agent = self.getAgentLocation()
        nobs = []
        for ap_idx, a_x, a_y, a_lvl in agent:
            nob = []
            food = self.get_neighbor_food(a_x, a_y)
            nob.extend(food)
            agentOrd = [ a_child for ac_idx, a_child in enumerate(agent) if ap_idx == ac_idx] + \
                       [ a_child for ac_idx, a_child in enumerate(agent) if ap_idx != ac_idx]
            for ac_idx, ac_x, ac_y, ac_lvl in agentOrd:
                position = self._transform_to_neighborhood((a_x, a_y), sight, (ac_x, ac_y))
                if min(position) < 0 or max(position) > 2*sight: 
                    nob.extend([-1, -1, 0])
                    continue 
                nob.extend(position); nob.append(ac_lvl)
            nobs.extend(nob.copy())
        return nobs 
    
    def __str__(self) -> str:
        p_str = ""
        for row in self.grid:
            row_str = ""
            for col in row:
                if col > 0: # agent
                    lvl, idx = int(col // num_agents), int(col % num_agents)
                    row_str += f'p_{idx}_{lvl}\t'
                elif col < 0: # food
                    row_str += f'f{-1*int(col)}\t'
                else: 
                    row_str += f'0\t'
            p_str += f"{row_str}\n"
        p_str += "\n"
        return p_str
    
###########################################################################################

class ParticleFilter:
    def __init__(self, nobs, nstate = [], J=100) -> None:
        self.greedy_epsilon = 0.95
        self.epsilon = 1e-9
        self.lr = 0.75
        self.J = J 
        self._X = []
        for state in nstate:
            self._X.append((1/J, state))
        for _ in range(J - len(nstate)):
            self._X.append((1/J, State(nobs))) # logits & state.

    def m_logits(self, x, u):
        xn = np.array(x)
        un = np.array(u); un = un.flatten()
        if xn.shape[0] != un.shape[0]: return self.epsilon
        match = 0
        for idx in range(xn.shape[0]):
            if idx % 3 == 2 and xn[idx] != un[idx]: return self.epsilon
            match += 1/(math.sqrt(1 + (xn[idx] - un[idx])**2)) # Coordinates.
        return match
 
    def normalize(self, logits):
        nu = np.exp(logits) 
        dn = sum(nu)
        n = len(logits)
        return [ nu[idx]/dn for idx in range(n) ]
    
    def decr(self):
        self.J = self.J // 10

    def particle_filter(self, action, nobs):
        Xn = []
        elem, prob = [], []
        for idx in range(self.J):
            self._X[idx][-1].move(action)
            z = self._X[idx][-1].nObs()
            wt = self.m_logits(z, nobs)
            elem.append(idx)
            prob.append(wt)
        prob = self.normalize(prob)
        for idx in range(int(self.J)):
            temp = self.lr * prob[idx] + (1 - self.lr) * self._X[idx][0]
            Xn.append((temp, self._X[idx][-1]))
        self._X = sorted(Xn, key = lambda x: -x[0])
        self._X = self._X[:int(self.J * self.greedy_epsilon)]
        min_p = min(prob)
        for idx in range(int(self.J - len(self._X))):
            self._X.append((min_p, State(nobs)))
        assert len(self._X) == self.J, f'Act[{len(self._X)}]; Exp[{self.J}]'

    @property
    def X(self):
        return self._X 

###########################################################################################

'''
    RUN ENV
'''
def test_particle_filter_env():
    def printPF():
        best_two_states = 1
        cnt = 0
        for state in pf.X:
            LOG.debug(f'Wt[{state[0]}]; \n{state[1]}')
            cnt += 1
            if cnt >= best_two_states: break
    jmax = 10000
    for episode in range(episodes):
        LOG.info(f'EPISODE {episode}')
        obs, info = env.reset()
        # LOG.debug(f'ENV Obs[{obs}]'); 
        env.printMap()
        pf = ParticleFilter(obs, J = jmax)
        LOG.debug(f'PF Obs ---------------'); printPF()
        LOG.debug(f'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        decr_times = 2
        for _ in range(episode_length):
            # if render_env:
            #     env.render()
            #     time.sleep(1) 
            action = [random.randint(0, num_actions-1) for __ in range(num_agents)]
            LOG.debug(f'-----------------\nAction[{action}]\n-----------------')
            ### action = USE NN to get action pf.X
            obs, reward, done, info = env.step(action)
            # LOG.debug(f'ENV Obs[{obs}]'); 
            env.printMap()
            pf.particle_filter(action, obs)
            LOG.debug(f'PF Obs ---------------'); printPF()
            LOG.debug(f'Reward[{reward}]-------------------')
            LOG.debug(f'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            # if _ < decr_times: pf.decr()
            if sum(done) == num_agents: break 
    env.close()

def test_particle_filter_sanity():
    obs = [[-1., -1.,  0., -1., -1.,  0.,  2.,  2.,  1., -1., -1.,  0.],\
           [ 2.,  1.,  1., -1., -1.,  0.,  2.,  2.,  1., -1., -1.,  0.]]
    pf = ParticleFilter(obs, J = 2)
    for state in pf.X:
        LOG.info(f'Wt[{state[0]}]; \n{state[1]}')
    assert True 

def test_particle_filter_food_agent():
    obs = [[-1., -1.,  0., -1., -1.,  0.,  2.,  2.,  1., -1., -1.,  0.],\
           [ 2.,  1.,  1., -1., -1.,  0.,  2.,  2.,  1., -1., -1.,  0.]]
    obj_state = State(obs)
    foods, agents = obj_state.foods, obj_state.agents
    for fidx, food in enumerate(foods):
        LOG.info(f'FOOD[{fidx}] {food}')
    for aidx, agent in enumerate(agents):
        LOG.info(f'AGENT[{aidx}] {agent}')
    assert True 

def test_particle_filter_gaussian():
    obs = [[-1., -1.,  0., -1., -1.,  0.,  2.,  2.,  1., -1., -1.,  0.],\
           [ 2.,  1.,  1., -1., -1.,  0.,  2.,  2.,  1., -1., -1.,  0.]]
    pf = ParticleFilter(obs, J = 2)
    LOG.debug(f'Prob. [{pf.gaussian([1, 2, 0, 1],[[1, 2], [0, 1]])}]')
    assert True 

def test_particle_filter_maintain_state():
    def printPF():
        best_two_states = 1
        cnt = 0
        for state in pf.X:
            LOG.debug(f'Wt[{state[0]}]; \n{state[1]}')
            cnt += 1
            if cnt >= best_two_states: break
    def convToState(state):
        for _ in range(grid_size):
            for __ in range(grid_size):
                if state[_][__] == '0':
                    state[_][__] = 0
                elif state[_][__][0] == 'f':
                    state[_][__] = -1 * int(state[_][__][1:])
                elif state[_][__][0] == 'p':
                    temp_list = state[_][__].split('_')
                    indx, lvl = map(int, temp_list[1:])
                    state[_][__] = lvl * num_agents + indx 
    obs, info = env.reset()
    actual_state = env.printMap()
    convToState(actual_state)
    pf = ParticleFilter(obs, [State(obs, actual_state, override = True)], J = 1000)
    LOG.debug(f'PF Obs ---------------'); printPF()
    LOG.debug(f'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    test_episode_length = 10
    for _ in range(test_episode_length):
            action = [random.randint(0, num_actions-1) for __ in range(num_agents)]
            LOG.debug(f'Action[{action}]-------------------')
            ### action = USE NN to get action pf.X
            obs, reward, done, info = env.step(action)
            env.printMap()
            pf.particle_filter(action, obs)
            LOG.debug(f'PF Obs ---------------'); printPF()
            LOG.debug(f'Reward[{reward}]-------------------')
            LOG.debug(f'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            if sum(done) == num_agents: break 

if __name__ == '__main__':
    #test_particle_filter_env()
    test_particle_filter_maintain_state()