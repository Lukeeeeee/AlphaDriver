import tensorflow as tf
import tensorlayer as tl
from src.model.critic.critic import Critic


class DenseCritic(Critic):
    def __init__(self, config, sess_flag=False, data=None):
        super(DenseCritic, self).__init__(config, sess_flag, data)

        self.net = self.create_model(state=self.state,
                                     action=self.action,
                                     name_prefix='CRITIC_')

        self.loss, self.optimizer = self.create_training_method(q_value=self.q_value,
                                                                action=self.action,
                                                                var_list=self.var_list)

        self.target_net = self.create_model(state=self.target_state,
                                            action=self.target_action,
                                            name_prefix='TARGET_CRITIC_')

    def create_model(self, state, action, name_prefix):
        state_net = tl.layers.InputLayer(inputs=state, name=name_prefix + 'CRITIC_STATE_INPUT_LAYER')
        action_net = tl.layers.InputLayer(inputs=action, name=name_prefix + 'CRITIC_ACTION_INPUT_LAYER')

        state_net = tl.layers.DenseLayer(layer=state_net,
                                         n_units=self.config.config_dict['STATE_LAYER_1_UNIT'],
                                         act=tf.nn.relu,
                                         name=name_prefix + 'STATE_DENSE_LAYER_1')
        action_net = tl.layers.DenseLayer(layer=action_net,
                                          n_units=self.config.config_dict['ACTION_LAYER_1_UNIT'],
                                          act=tf.nn.relu,
                                          name=name_prefix + 'ACTION_DENSE_LAYER_1')
        net = tf.stack(values=[state_net.outputs, action_net.outputs], axis=1)
        net = tf.reshape(tensor=net,
                         shape=[-1, self.config.config_dict['ACTION_LAYER_1_UNIT'] +
                                self.config.config_dict['STATE_LAYER_1_UNIT']])

        net = tl.layers.InputLayer(inputs=net,
                                   name=name_prefix + 'MIDDLE_INPUT')

        net = tl.layers.DenseLayer(layer=net,
                                   n_units=self.config.config_dict['MERGED_LAYER_1_UNIT'],
                                   act=tf.nn.relu,
                                   name=name_prefix + 'MERGED_DENSE_LAYER_1')

        net = tl.layers.DenseLayer(layer=net,
                                   n_units=self.config.config_dict['OUTPUT_LAYER'],
                                   act=tf.nn.relu,
                                   name=name_prefix + 'OUTPUT_LAYER')
        return net

    def create_training_method(self, q_value, action, var_list):
        weight_decay = tf.add_n([self.config.config_dict['L2'] * tf.nn.l2_loss(var) for var in self.var_list])
        loss = tf.reduce_mean(tf.square(self.q_label - self.q_value)) + weight_decay
        optimizer = tf.train.AdamOptimizer(self.config.config_dict['LEARNING_RATE'])
        return loss, optimizer

    def update(self, sess, q_label, state, action):
        loss, _, grad = sess.run(fetches=[self.loss, self.optimize_loss, self.gradients],
                                 feed_dict={
                                     self.q_label: q_label,
                                     self.state: state,
                                     self.action: action,
                                     self.is_training: True
                                 })
        return loss, grad

    def eval_tensor(self, ):
        raise NotImplementedError
        pass


if __name__ == '__main__':
    from src.config.config import Config
    from src.config.utils import load_json
    from configuration import CONFIG_PATH
    from configuration.standard_key_list import CONFIG_STANDARD_KEY_LIST

    key_list = load_json(file_path=CONFIG_STANDARD_KEY_LIST + '/criticKeyList.json')
    a = Config(standard_key_list=key_list)
    a.load_config(path=CONFIG_PATH + '/testCriticConfig.json')
    critic = DenseCritic(config=a)
