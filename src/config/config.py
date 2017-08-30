from src.config.utils import load_json, save_to_json, check_dict_key


class Config(object):
    def __init__(self, standard_key_list, config_dict=None):
        self.standard_key_list = standard_key_list

        if config_dict:
            self._config_dict = config_dict
        else:
            self._config_dict = {}

    @property
    def config_dict(self):
        return self._config_dict

    @config_dict.setter
    def config_dict(self, new_value):
        if self.check_config(dict=new_value, key_list=self.standard_key_list) is True:
            self._config_dict = new_value

    def save_config(self, path):
        save_to_json(dict=self.config_dict, path=path)

    def load_config(self, path):
        res = load_json(file_path=path)
        self.config_dict = res

    def check_config(self, dict, key_list):
        if check_dict_key(dict=dict, standard_key_list=key_list):
            return True
        else:
            return False


if __name__ == '__main__':
    config = {
        'A': 1,
        'B': 2
    }
    key_list = ['A', 'B', 'C']
    c = Config(config_dict=config, standard_key_list=key_list)
