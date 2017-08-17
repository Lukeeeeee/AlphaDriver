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

        self.action, self.var_list, self.keep_prob = self.create_model(state=self.state)
        self.target_action, self.target_var_list, _ = self.create_model(state=self.target_state)
        self.optimizer, self.optimize_loss = self.create_training_method()

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

        action = net.outputs
        keep_prob = net.all_drop
        var_list = net.all_params

        return action, var_list, keep_prob

    def create_training_method(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config['LEARNING_RATE'])
        optimize_loss = optimizer.apply_gradients(grads_and_vars=self.gradients)
        return optimizer, optimize_loss

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

