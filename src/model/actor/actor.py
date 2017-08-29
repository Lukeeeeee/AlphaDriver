from src.model.model import Model
import tensorlayer as tl
import tensorflow as tf
import src.model.utils as utils


class Actor(Model):
    def __init__(self, config, sess_flag=False, data=None):
        super(Actor, self).__init__(config, sess_flag, data)

        self.state = tf.placeholder(tf.float32, shape=[None, self.config['STATE_DIM']])
        self.is_training = tf.placeholder(tf.bool)
        self.gradients = tf.placeholder(tf.float32, [None, self.config['ACTION_DIM']])

        self.target_state = tf.placeholder(tf.float32, shape=[None, self.config['STATE_DIM']])
        self.target_is_training = tf.placeholder(tf.bool)

        self.net = None
        self.target_net = None

        self.loss = None
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
    def optimize_loss(self):
        return self.optimizer.minimize(self.loss)

    @property
    def target_action(self):
        return self.target_net.outputs

    @property
    def target_var_list(self):
        return self.target_net.all_params

    def create_model(self, state):
        pass

    def create_training_method(self):
        pass

    def update(self, sess, gradients, state):
        super(Actor, self).update()
        loss, gradients = sess.run(fetches=[self.optimize_loss, self.gradients],
                                   feed_dict={
                                       self.gradients: gradients,
                                       self.state: state,
                                       self.is_training: True
                                    })
        return loss, gradients

    def update_target(self):
        # TODO
        # USE tensor.assign

        for var, target_var in zip(self.var_list, self.target_var_list):
            target_var = self.config['DECAY'] * var + (1.0 - self.config['DECAY']) * target_var

    def predict(self, sess, state):
        action = sess.run(fetches=[self.action],
                          feed_dict={
                              self.state: state,
                              self.is_training: False
                          })
        return action

    def predict_target(self, sess, state):
        action = sess.run(fetches=[self.target_action],
                          feed_dict={
                              self.target_state: state,
                              self.target_is_training: False
                          })
        return action

    def eval_tensor(self, ):
        raise NotImplementedError
        pass

