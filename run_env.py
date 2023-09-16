import gym
import rware
import time
import random
import logging  
LOG = logging.getLogger(__name__)
logging.basicConfig()
LOG.setLevel(logging.INFO)

render_env= True 
num_actions, num_agents = 5, 2 
env = gym.make(f"rware-tiny-{num_agents}ag-v1")
episodes, episode_length = 1, 100 
for episode in range(episodes):
    LOG.info(f'EPISODE {episode}')
    obs = env.reset()
    for step in range(episode_length):
        if render_env:
            env.render()
            time.sleep(1)
        action = tuple(random.randint(0,num_actions-1) for _ in range(num_agents))
        obs, reward, done, info = env.step(action)
        LOG.info(f'Action[{action}] Reward[{reward}] Obs[{obs}]')
        if sum(done) == num_agents: break 
env.close()