from src.environment.environment import Environment
from src.environment.torcsEnvironment.torcs.gym_torcs import TorcsEnv
from collections import deque
from src.environment.utils import slice_queue
import src.environment.utils as utils
from configuration.standard_key_list import CONFIG_STANDARD_KEY_LIST


class TorcsEnvironment(Environment):
    standard_key_list = utils.load_json(CONFIG_STANDARD_KEY_LIST + '/torcsEnvironmentKeyList.json')

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
        img = self.return_latest_state_from_vision_list()
        return img, reward, done

    def reset(self, relauch=False):
        ob = self.torcs_env.reset(relaunch=relauch)
        self.torcs_vision_list.clear()
        while len(self.torcs_vision_list) < self.config.config_dict['VISION_DATA_LENGTH']:
            self.torcs_vision_list.put(ob.img)
        img = self.return_latest_state_from_vision_list()
        return img

    def end(self):
        self.torcs_env.end()
        self.torcs_vision_list.clear()

    def return_latest_state_from_vision_list(self):
        right = len(self.torcs_vision_list) - 1
        left = right - self.config.config_dict['VISION_DATA_LENGTH']
        img = slice_queue(q=self.torcs_vision_list,
                          left=left,
                          right=right)
        return img


if __name__ == '__main__':
    from src.config.config import Config
    from configuration import CONFIG_PATH
    a = Config(standard_key_list=TorcsEnvironment.standard_key_list)
    a.load_config(path=CONFIG_PATH + '/testTorcsEnvironmentConfig.json')
    env = TorcsEnvironment(config=a)
