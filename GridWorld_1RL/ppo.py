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

def run_env(s: str):
    print(s)

G = GLOBAL()
for episode in range(G.episodes):
    proc_arr = []
    for agent in range(G.n_agents):
        process = mpc.Process(target = run_env, args = (f"agent_{agent}",))
        process.start()
        proc_arr.append(process)
    for agent in range(G.n_agents):
        proc_arr[agent].join()
