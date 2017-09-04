import tensorlayer as tl
import tensorflow as tf
import src.model.utils as utils
from src.model.actor.actor import Actor


class DenseActor(Actor):
    def __init__(self, config, sess_flag=False, data=None):
        super(DenseActor, self).__init__(config, sess_flag, data)

        self.net = self.create_model(self.state, 'ACTOR_')
        self.target_net = self.create_model(self.target_state, 'TARGET_ACTOR_')
        self.optimizer, self.optimize_loss = self.create_training_method()

    def create_model(self, state, name_prefix):
        net = tl.layers.InputLayer(inputs=state,
                                   name=name_prefix + 'INPUT_LAYER')
        net = tl.layers.DenseLayer(layer=net,
                                   n_units=self.config.config_dict['DENSE_LAYER_1_UNIT'],
                                   act=tf.nn.relu,
                                   name=name_prefix + 'DENSE_LAYER_1')
        net = tl.layers.DropconnectDenseLayer(layer=net,
                                              n_units=self.config.config_dict['DENSE_LAYER_2_UNIT'],
                                              act=tf.nn.relu,
                                              name=name_prefix + 'DENSE_LAYER_2',
                                              keep=self.config.config_dict['DROP_OUT_PROB_VALUE'])
        net = tl.layers.DenseLayer(layer=net,
                                   n_units=self.config.config_dict['ACTION_DIM'],
                                   act=tf.nn.tanh,
                                   name=name_prefix + 'OUTPUT_LAYER')

        return net

    def create_training_method(self):
        parameters_gradients = tf.gradients(self.action, self.var_list, -self.q_value_gradients)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.config_dict['LEARNING_RATE'])
        optimize_loss = optimizer.apply_gradients(grads_and_vars=zip(parameters_gradients, self.var_list))
        return optimizer, optimize_loss


if __name__ == '__main__':
    from src.config.config import Config
    from configuration import CONFIG_PATH
    from configuration.standard_key_list import CONFIG_STANDARD_KEY_LIST
    from src.config.utils import load_json

    key_list = load_json(file_path=CONFIG_STANDARD_KEY_LIST + '/actorKeyList.json')
    a = Config(config_dict=None, standard_key_list=key_list)
    a.load_config(path=CONFIG_PATH + '/testActorConfig.json')
    actor = DenseActor(config=a)
    pass
