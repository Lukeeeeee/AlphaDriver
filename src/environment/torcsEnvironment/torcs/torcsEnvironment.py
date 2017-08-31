from src.environment.environment import Environment
from src.environment.torcsEnvironment.torcs.gym_torcs import TorcsEnv
from collections import deque
from src.environment.utils import slice_queue


class TorcsEnvironment(Environment):
    def __init__(self, config):
        super(TorcsEnvironment, self).__init__(config)
        self.torcs_env = TorcsEnv(vision=True,
                                  throttle=True,
                                  gear_change=False)
        self.torcs_vision_list = deque(maxlen=self.config.config_dict['VISION_MAXLEN'])

    def step(self, action):
        ob, reward, done, info = self.torcs_env.step(u=action)
        self.torcs_vision_list.put(ob.img)
        while len(self.torcs_vision_list) < self.config.config_dict['VISION_DATA_LENGTH']:
            self.torcs_vision_list.put(ob.img)
        right = len(self.torcs_vision_list) - 1
        left = right - self.config.config_dict['VISION_DATA_LENGTH']
        img = slice_queue(q=self.torcs_vision_list,
                          left=left,
                          right=right)
        return img, reward, done

    def reset(self, relauch=False):
        self.torcs_env.reset(relaunch=relauch)

    def end(self):
        self.torcs_env.end()
