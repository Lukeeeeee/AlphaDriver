

class Agent(object):

    standard_key_list = []

    def __init__(self, env, config, model=None):
        self.env = env
        self.config = config
        self.model = model
        self.state = None
        self.action = None
        self.reward = None
        self.reward_list = None
        pass

    def observe(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def play(self, *args, **kwargs):
        pass

