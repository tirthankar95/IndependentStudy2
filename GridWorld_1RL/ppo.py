from ENV.gridWorld import *
import multiprocessing as mpc
from UTILS.global_tm import Global
from POLICY.policy import Policy
from NN_ARCH.nn_model import NNet
from UTILS import helper as hp

def run_env(s: str):
    state_arr, next_state_arr, advantage_arr, reward_arr, action_arr = [], [], [], [], []
    gB = GridWorldBuilder()
    obs = gB.reset() 
    obs = hp.convert(obs)
    while True:
        action = p.get_action(obs)
        nobs, reward, terminated, truncated = gB.step(action)
        nobs = hp.convert(nobs)
        # Store Data.
        state_arr.append(obs)
        next_state_arr.append(nobs)
        reward_arr.append(reward)
        action_arr.append(action.value)
        if terminated or truncated:
            break
        obs = nobs

G = Global()
p = Policy(NNet(hp.get_init_data(GridWorldBuilder()), Action.MX_ACTION.value))
for episode in range(G.episodes):
    proc_arr = []
    for agent in range(G.n_agents):
        process = mpc.Process(target = run_env, args = (f"agent_{agent}",))
        process.start()
        proc_arr.append(process)
    for agent in range(G.n_agents):
        proc_arr[agent].join()
    p.update_policy()

# TBD 
# 1. Actor Critic NN. [DONE]
# 2. Complete PPO in parallel form
# 3. Notice PPO solve the environment.
# -------------------------------------- 
# 4. Create multi-agent grid world. -> Object collection.
#    Check lb-foraging environment and see how to do it.
# 5. Update PPO algorithm for multi-agent.
# 6. Check if agent is able to solve it.
# ---------------------------------------
# 7. Partial observation agent performance. o1,o2,o3 ... Some attention models.
# 8. Agent is given state estimates at the beginning s + o1 + o2 ... how does it change.
# 9. Add LSTM to NN.