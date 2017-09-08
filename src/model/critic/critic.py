from src.model.model import Model
import tensorflow as tf
import tensorlayer as tl


class Critic(Model):
    def __init__(self, config, sess_flag=False, data=None):
        super(Critic, self).__init__(config, sess_flag, data)

        # TODO
        # HOW TO DEFINE THE TYPE OF THE STATE_DIM LIST OR A SCALAR?
        if type(self.config.config_dict['STATE_DIM']) is list:
            self.state = tf.placeholder(tf.float32, shape=[None] + self.config.config_dict['STATE_DIM'])
            self.target_state = tf.placeholder(tf.float32, shape=[None] + self.config.config_dict['STATE_DIM'])
        else:
            self.state = tf.placeholder(tf.float32, shape=[None, self.config.config_dict['STATE_DIM']])
            self.target_state = tf.placeholder(tf.float32, shape=[None, self.config.config_dict['STATE_DIM']])

        if type(self.config.config_dict['ACTION_DIM']) is list:
            self.action = tf.placeholder(tf.float32, shape=[None] + self.config.config_dict['ACTION_DIM'])
            self.target_action = tf.placeholder(tf.float32, shape=[None] + self.config.config_dict['ACTION_DIM'])
        else:
            self.action = tf.placeholder(tf.float32, shape=[None, self.config.config_dict['ACTION_DIM']])
            self.target_action = tf.placeholder(tf.float32, shape=[None, self.config.config_dict['ACTION_DIM']])

        self.q_label = tf.placeholder(tf.float32, shape=[None, 1])
        self.is_training = tf.placeholder(tf.bool)

        self.target_is_training = tf.placeholder(tf.bool)

        self.net = None

        self.optimizer = None
        self.loss = None

        self.target_net = None

    @property
    def q_value(self):
        return self.net.outputs

    @property
    def var_list(self):
        return self.net.all_params

    @property
    def keep_prob(self):
        return self.net.all_drop

    @property
    def optimize_loss(self):
        return self.optimizer.minimize(self.loss)

    @property
    def action_gradients(self):
        return tf.gradients(self.q_value, self.action)

    @property
    def gradients(self):
        gradients = self.optimizer.compute_action_gradients(self.loss, var_list=self.var_list)
        return gradients

    @property
    def target_q_value(self):
        return self.target_net.outputs

    @property
    def target_var_list(self):
        return self.target_net.all_params

    def create_model(self, state, action, name_prefix):
        pass

    def create_training_method(self, q_value, action, var_list):
        pass

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
            target_var = self.config.config_dict['DECAY'] * var + (1.0 - self.config.config_dict['DECAY']) * target_var

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

    def compute_action_gradients(self, sess, state, action):
        res = sess.run(fetches=[self.action_gradients],
                       feed_dict={
                           self.state: state,
                           self.action: action,
                           self.is_training: False
                       })
        return res

    def eval_tensor(self, ):
        raise NotImplementedError
        pass

