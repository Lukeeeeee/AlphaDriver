

class Environment(object):
    standard_key_list = []
    
    def __init__(self, config):
        self.config = config
        pass

    def step(self, *args, **kwargs):
        return None, None, None

    def reset(self, *args, **kwargs):
        return None

    def end(self, *args, **kwargs):
        pass
