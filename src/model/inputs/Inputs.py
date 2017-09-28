import tensorflow as tf


class Inputs(object):

    def __init__(self, config):
        self.input_dict = {}
        for key, value in config.items():
            if type(value) is list:
                self.input_dict[key] = tf.placeholder(tf.float32, shape=[None] + value)
            elif type(value) is int:
                self.input_dict[key] = tf.placeholder(tf.float32, shape=[None, value])
            else:
                raise TypeError('does not support %s to init a input tensor' % str(type(value)))

        pass

