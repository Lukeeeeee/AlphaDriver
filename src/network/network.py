import tensorflow as tf
import tensorlayer as tl


class Network(object):
    def __init__(self, input, config):
        self.input = input
        self.config = config
        with tf.name_scope(self.config['NAME']):
            net = tl.layers.InputLayer(inputs=input, name='INPUT_LAYER')
            net = tl.layers.DenseLayer(layer=net,
                                       n_units=self.config['DENSE_LAYER_1_UNIT'],
                                       act=tf.nn.relu,
                                       name='DENSE_LAYER_1')
            net = tl.layers.DropconnectDenseLayer(layer=net,
                                                  n_units=self.config['DENSE_LAYER_2_UNIT'],
                                                  act=tf.nn.relu,
                                                  name='DENSE_LAYER_2',
                                                  keep=0.5)
            net = tl.layers.DenseLayer(layer=net,
                                       n_units=self.config['ACTION_DIM'],
                                       act=tf.nn.tanh,
                                       name='OUTPUT_LAYER')
            self.output = net.outputs
            self.keep_prob = net.all_drop
            self.var_list = net.all_params
