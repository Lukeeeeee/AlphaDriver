from src.config.config import Config
from src.config.utils import check_dict_key, load_json
from configuration.standard_key_list import CONFIG_STANDARD_KEY_LIST


class DDPGConfig(Config):
    def __init__(self, standard_key_list, config_path=None):
        super(DDPGConfig, self).__init__(standard_key_list=standard_key_list)
        actor_key_list = load_json(file_path=CONFIG_STANDARD_KEY_LIST + '/actorKeyList.json')
        self.actor_config = Config(standard_key_list=actor_key_list)

        critic_key_list = load_json(file_path=CONFIG_STANDARD_KEY_LIST + '/criticKeyList.json')
        self.critic_config = Config(standard_key_list=critic_key_list)
        if config_path:
            self.load_config(path=config_path)

    @property
    def config_dict(self):
        return super(DDPGConfig, self).config_dict

    @config_dict.setter
    def config_dict(self, new_value):
        if check_dict_key(dict=new_value, standard_key_list=self.standard_key_list) is True:
            self._config_dict = new_value
            self.actor_config.config_dict = new_value['ACTOR_CONFIG']
            self.critic_config.config_dict = new_value['CRITIC_CONFIG']
        pass


if __name__ == '__main__':
    from configuration import CONFIG_PATH
    from configuration.standard_key_list import CONFIG_STANDARD_KEY_LIST

    key_list = load_json(file_path=CONFIG_STANDARD_KEY_LIST + '/ddpgKeyList.json')
    a = DDPGConfig(standard_key_list=key_list,
                   config_path=CONFIG_PATH + '/testDDPGConfig.json')
    pass
