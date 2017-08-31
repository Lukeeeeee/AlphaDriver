from src.environment.environment import Environment
from src.environment.torcsEnvironment.torcs.gym_torcs import TorcsEnv


class TorcsEnvironment(Environment):
    def __init__(self):
        super(TorcsEnvironment, self).__init__()
        self.torcs_env = TorcsEnv(vision=False,
                                  throttle=True,
                                  gear_change=False)

    def step(self, action):
        ob, reward, done, info = self.torcs_env.step(u=action)
        pass
