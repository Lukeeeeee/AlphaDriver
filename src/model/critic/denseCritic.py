from src.model.model import Model
import tensorflow as tf
import tensorlayer as tl
from src.model.critic.critic import Critic


class DenseCritic(Critic):
    def __init__(self, config, sess_flag=False, data=None):
        super(DenseCritic, self).__init__(config, sess_flag, data)

        self.net = self.create_model(state=self.state, action=self.action)

        self.loss, self.optimizer = self.create_training_method(q_value=self.q_value,
                                                                action=self.action,
                                                                var_list=self.var_list)

        self.target_net = self.create_model(state=self.target_state,
                                            action=self.target_action)

    def create_model(self, state, action):
        state_net = tl.layers.InputLayer(inputs=state, name='CRITIC_STATE_INPUT_LAYER')
        action_net = tl.layers.InputLayer(inputs=action, name='CRITIC_ACTION_INPUT_LAYER')

        state_net = tl.layers.DenseLayer(layer=state_net,
                                         n_units=self.config['STATE_LAYER_1_UNIT'],
                                         act=tf.nn.relu,
                                         name='STATE_DENSE_LAYER_1')
        action_net = tl.layers.DenseLayer(layer=action_net,
                                          n_units=self.config['ACTION_LAYER_1_UNIT'],
                                          act=tf.nn.relu,
                                          name='ACTION_DENSE_LAYER_1')
        net = tf.stack(values=[state_net.outputs, action_net.outputs], axis=0)

        net = tl.layers.DenseLayer(layer=net,
                                   n_units=self.config['MERGED_LAYER_1_UNIT'],
                                   act=tf.nn.relu,
                                   name='MERGED_DENSE_LAYER_1')

        net = tl.layers.DenseLayer(layer=net,
                                   n_units=self.config['MERGED_LAYER_2_UNIT'],
                                   act=tf.nn.relu,
                                   name='MERGED_DENSE_LAYER_2')
        return net

    def create_training_method(self, q_value, action, var_list):
        weight_decay = tf.add_n([self.config['L2'] * tf.nn.l2_loss(var) for var in self.var_list])
        loss = tf.reduce_mean(tf.square(self.q_label - self.q_value)) + weight_decay
        optimizer = tf.train.AdamOptimizer(self.config['LEARNING_RATE'])
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
