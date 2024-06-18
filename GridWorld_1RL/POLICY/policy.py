from ENV.gridWorld import *

class Policy:
    def __init__(self, nn) -> None:
        self.nn = nn 
    def get_action(self, obs) -> Action:
        mu_dist, value = self.nn(obs)
        return Action(mu_dist.sample().item())
    def update_policy(self):
        pass