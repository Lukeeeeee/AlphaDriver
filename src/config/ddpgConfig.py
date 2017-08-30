from configuration.standard_key_list import criticKeyList, actorKeyList
from src.config.config import Config
from src.config.utils import check_dict_key


class DDPGConfig(Config):
    def __init__(self, standard_key_list, config_path=None):
        super(DDPGConfig, self).__init__(standard_key_list=standard_key_list)
        self.actor_config = Config(standard_key_list=actorKeyList.key_list)

        self.critic_config = Config(standard_key_list=criticKeyList.key_list)
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
    from configuration.standard_key_list import ddpgKeyList

    a = DDPGConfig(standard_key_list=ddpgKeyList.key_list,
                   config_path=CONFIG_PATH + '/testDDPGConfig.json')
    pass
