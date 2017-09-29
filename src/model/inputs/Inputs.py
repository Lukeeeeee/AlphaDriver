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
        self.tensor_tuple = tuple(value for _, value in self.input_dict.items())

    def generate_inputs_tuple(self, data_dict):
        res = tuple(data_dict[str(key)] for key, _ in self.input_dict.items())
        return res

    def __call__(self, name=None):
        if name is not None:
            return self.input_dict[name]
        else:
            return self.input_dict


if __name__ == '__main__':
    config = {
        'IMAGE': [1, 2, 3],
        'SPEED': [3],
        'POS': [1]
    }

    a = Inputs(config)
    print a.tensor_tuple
    print a.generate_inputs_tuple(data_dict={'IMAGE': 0,
                                             'SPEED': 1,
                                             'POS': 2})
    print a()
    print a('IMAGE')

