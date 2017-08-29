import tensorflow as tf


class Model(object):

    def __init__(self, config, sess_flag=False, data=None):
        self.config = config
        self.data = data
        self.net = None
        if sess_flag is True:
            self.sess = tf.InteractiveSession()

    def create_model(self, *args, **kwargs):
        pass

    def create_training_method(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def eval_tensor(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass

    def save_model(self, *args, **kwargs):
        pass

    def load_model(self, *args, **kwargs):
        pass
