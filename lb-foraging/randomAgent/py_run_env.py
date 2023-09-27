import gym 
import sys 
sys.path.append('../')
import lbforaging 
import random
import time
import logging 
LOG = logging.getLogger(__name__)
logging.basicConfig()
LOG.setLevel(logging.INFO)

render_env= False 
num_actions, num_agents = 6, 2
episodes, episode_length = 1, 10
'''
    Foraging-{GRID_SIZE}x{GRID_SIZE}-{PLAYER COUNT}p-{FOOD LOCATIONS}f{-coop IF COOPERATIVE MODE}-v0
'''
env = gym.make(f"Foraging-8x8-{num_agents}p-1f-v2")
for episode in range(episodes):
    LOG.info(f'EPISODE {episode}')
    obs = env.reset()
    for _ in range(episode_length):
        '''
        class Action(Enum):
            NONE = 0
            NORTH = 1
            SOUTH = 2
            WEST = 3
            EAST = 4
            LOAD = 5
        '''
        if render_env:
            env.render()
            time.sleep(1)
        action = [random.randint(0, num_actions-1) for __ in range(num_agents)]
        obs, reward, done, info = env.step(action)
        LOG.info(f'Action[{action}] Reward[{reward}] Obs[{obs}]')
        if sum(done) == num_agents: break 
env.close()