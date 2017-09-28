from src.model.model import Model
import tensorflow as tf
from src.model.inputs.Inputs import Inputs


class Actor(Model):
    def __init__(self, config, sess_flag=False, data=None):
        super(Actor, self).__init__(config, sess_flag, data)
        self.state = Inputs(config=self.config.config_dict['STATE'])
        self.target_state = Inputs(config=self.config.config_dict['STATE'])

        # if type(self.config.config_dict['STATE_DIM']) is list:
        #     self.state = tf.placeholder(tf.float32, shape=[None] + self.config.config_dict['STATE_DIM'])
        #     self.target_state = tf.placeholder(tf.float32, shape=[None] + self.config.config_dict['STATE_DIM'])
        # else:
        #     self.state = tf.placeholder(tf.float32, shape=[None, self.config.config_dict['STATE_DIM']])
        #     self.target_state = tf.placeholder(tf.float32, shape=[None, self.config.config_dict['STATE_DIM']])

        self.is_training = tf.placeholder(tf.bool)
        self.q_value_gradients = tf.placeholder(tf.float32, [None, self.config.config_dict['ACTION_DIM']])

        self.target_is_training = tf.placeholder(tf.bool)

        self.net = None
        self.target_net = None

        self.optimize_loss = None
        self.optimizer = None

    @property
    def action(self):
        return self.net.outputs

    @property
    def var_list(self):
        return self.net.all_params

    @property
    def keep_prob(self):
        return self.net.all_drop

    @property
    def target_action(self):
        return self.target_net.outputs

    @property
    def target_var_list(self):
        return self.target_net.all_params

    def create_model(self, state, name_prefix):
        pass

    def create_training_method(self):
        pass

    def update(self, sess, gradients, state):
        super(Actor, self).update()
        loss, gradients = sess.run(fetches=[self.optimize_loss, self.q_value_gradients],
                                   feed_dict={
                                       self.q_value_gradients: gradients,
                                       self.state.tensor_tuple: self.state.generate_inputs_tuple(data_dict=state),
                                       self.is_training: True
                                    })
        return loss, gradients

    def update_target(self):
        # TODO
        # USE tensor.assign

        for var, target_var in zip(self.var_list, self.target_var_list):
            target_var = self.config.config_dict['DECAY'] * var + (1.0 - self.config.config_dict['DECAY']) * target_var

    def predict(self, sess, state):
        action = sess.run(fetches=[self.action],
                          feed_dict={
                              self.state.tensor_tuple: self.state.generate_inputs_tuple(data_dict=state),
                              self.is_training: False
                          })
        return action

    def predict_target(self, sess, state):
        action = sess.run(fetches=[self.target_action],
                          feed_dict={
                              self.target_state.tensor_tuple: self.target_state.generate_inputs_tuple(data_dict=state),
                              self.target_is_training: False
                          })
        return action

    def eval_tensor(self, ):
        raise NotImplementedError
        pass

