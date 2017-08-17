from src.model.model import Model
import tensorflow as tf
import tensorlayer as tl


class Critic(Model):
    def __init__(self, config, sess_flag=False, data=None):
        super(Critic, self).__init__(config, sess_flag, data)

        # TODO
        # HOW TO DEFINE THE TYPE OF THE STATE_DIM LIST OR A SCALAR?

        self.state = tf.placeholder(tf.float32, shape=[None, self.config['STATE_DIM']])
        self.action = tf.placeholder(tf.float32, shape=[None, self.config['ACTION_DIM']])
        self.q_label = tf.placeholder(tf.float32, shape=[None, 1])
        self.is_training = tf.placeholder(tf.bool)
        self.q_value, self.var_list, self.keep_prob = self.create_model(state=self.state,
                                                                        action=self.action)
        self.loss, self.optimize_loss, self.action_gradients, \
        self.gradients = self.create_training_method(q_value=self.q_value,
                                                     action=self.action,
                                                     var_list=self.var_list)

        self.target_state = tf.placeholder(tf.float32, shape=[None, self.config['STATE_DIM']])
        self.target_action = tf.placeholder(tf.float32, shape=[None, self.config['ACTION_DIM']])
        self.target_is_training = tf.placeholder(tf.bool)

        self.target_q_value, self.target_var_list, _ = self.create_model(state=self.target_state,
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
        return net.outputs, net.all_params, net.all_drop

    def create_training_method(self, q_value, action, var_list):
        weight_decay = tf.add_n([self.config['L2'] * tf.nn.l2_loss(var) for var in self.var_list])
        loss = tf.reduce_mean(tf.square(self.q_label - self.q_value)) + weight_decay
        optimizer = tf.train.AdamOptimizer(self.config['LEARNING_RATE'])
        optimize_loss = optimizer.minimize(loss)
        gradients = optimizer.compute_gradients(loss, var_list=var_list)
        action_gradients = tf.gradients(q_value, action)
        return loss, optimize_loss, action_gradients, gradients

    def update(self, sess, q_label, state, action):
        loss, _, grad = sess.run(fetches=[self.loss, self.optimize_loss, self.gradients],
                                 feed_dict={
                                     self.q_label: q_label,
                                     self.state: state,
                                     self.action: action,
                                     self.is_training: True
                                 })
        return loss, grad

    def update_target(self):
        # TODO
        # USE tensor.assign

        for var, target_var in zip(self.var_list, self.target_var_list):
            target_var = self.config['DECAY'] * var + (1.0 - self.config['DECAY']) * target_var

    def predict(self, sess, state, action):
        q_value = sess.run(fetches=[self.q_value],
                           feed_dict={
                               self.action: action,
                               self.state: state,
                               self.is_training: False
                           })
        return q_value

    def predict_target(self, sess, state, action):
        q_value = sess.run(fetches=[self.target_q_value],
                           feed_dict={
                               self.target_action: action,
                               self.target_state: state,
                               self.target_is_training: False
                           })
        return q_value

    def eval_tensor(self, ):
        raise NotImplementedError
        pass

