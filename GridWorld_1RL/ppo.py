from gridWorld import *
import multiprocessing as mpc

# N = agents
# for episode in [1..T]
#     for agents in [1..N] -> in parallel 
#         \pi{old}
#     \pi{new} <- update()

class GLOBAL:
    episodes = 100
    n_agents = 8

def policy(obs):
    pass

def run_env(s: str):
    gB = GridWorldBuilder()
    obs = gB.reset()
    while True:
        action = policy(obs)
        obs, reward, terminated = gB.step(action)
        if terminated:
            break

G = GLOBAL()
for episode in range(G.episodes):
    proc_arr = []
    for agent in range(G.n_agents):
        process = mpc.Process(target = run_env, args = (f"agent_{agent}",))
        process.start()
        proc_arr.append(process)
    for agent in range(G.n_agents):
        proc_arr[agent].join()


# TBD 
# 1. Actor Critic NN.
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
