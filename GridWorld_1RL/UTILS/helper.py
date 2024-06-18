from ENV.gridWorld import * 
import torch 

def convert(obs, convert_var = True):
    n = len(obs)
    if not convert_var: return obs
    obs = [[obs[i][j].value for j in range(n)] \
                            for i  in range(n)]
    obs = torch.tensor(obs, dtype = torch.float32)
    obs = obs.unsqueeze(0).unsqueeze(0)
    return obs

def get_init_data(env):
    obs = env.reset()
    return convert(obs)
