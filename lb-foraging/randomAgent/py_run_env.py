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
'''
ENVIRONMENT PARAMS.
'''
grid_size, num_agents, num_food = 8, 2, 2
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
'''
    Foraging-{GRID_SIZE}x{GRID_SIZE}-{PLAYER COUNT}p-{FOOD LOCATIONS}f{-coop IF COOPERATIVE MODE}-v0
'''
env = gym.make(f"Foraging-{sight}s-{grid_size}x{grid_size}-{num_agents}p-{num_food}f-v2")


for episode in range(episodes):
    LOG.info(f'EPISODE {episode}')
    obs = env.reset()
    print(f'Obs[{obs}]')
    for _ in range(episode_length):
        if render_env:
            env.render()
            time.sleep(1)
        action = [random.randint(0, num_actions-1) for __ in range(num_agents)]
        obs, reward, done, info = env.step(action)
        LOG.info(f'Action[{action}] Reward[{reward}] Obs[{obs}]')
        env.printMap()
        if sum(done) == num_agents: break 
env.close()