from src.model.model import Model
import tensorlayer as tl
import tensorflow as tf
import src.model.utils as utils
from src.model.actor.actor import Actor


class DenseActor(Actor):
    def __init__(self, config, sess_flag=False, data=None):
        super(DenseActor, self).__init__(config, sess_flag, data)

        self.net = self.create_model(self.state)
        self.target_net = self.create_model(self.target_state)
        self.loss, self.optimizer = self.create_training_method()

    def create_model(self, state):
        net = tl.layers.InputLayer(inputs=state, name='ACTOR_INPUT_LAYER')
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
        # TODO
        # ADD DIFFERENT ACT FUNCTION FOR OUTPUT VAR
        # CURRENT IS ALL TANH
        return net

    def create_training_method(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config['LEARNING_RATE'])
        optimize_loss = optimizer.apply_gradients(grads_and_vars=self.gradients)
        return optimizer, optimize_loss
