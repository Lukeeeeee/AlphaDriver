from src.model.critic.critic import Critic
import tensorflow as tf
import tensorlayer as tl
import tensorflow.contrib.layers as tf_contrib_layers


class LSTMCritic(Critic):
    def __init__(self, config, sess_flag=False, data=None):
        super(LSTMCritic, self).__init__(config, sess_flag, data)
        self.net = self.create_model(state=self.state,
                                     action=self.action,
                                     name_prefix='CRITIC_')
        self.target_net = self.create_model(state=self.target_state,
                                            action=self.action,
                                            name_prefix='TARGET_CRITIC_')

    def create_model(self, state, action, name_prefix):

        # Create aciton net
        action_net = tl.layers.InputLayer(inputs=action, name=name_prefix + 'CRITIC_ACTION_INPUT_LAYER')

        action_net = tl.layers.DenseLayer(layer=action_net,
                                          n_units=self.config.config_dict['ACTION_LAYER_1_UNIT'],
                                          act=tf.nn.relu,
                                          name=name_prefix + 'ACTION_DENSE_LAYER_1')
        action_net = tl.layers.DenseLayer(layer=action_net,
                                          n_units=self.config.config_dict['ACTION_LAYER_2_UNIT'],
                                          act=tf.nn.relu,
                                          name=name_prefix + 'ACTION_DENSE_LAYER_2')

        # Create state lstm cnn net

        W_init = tf.truncated_normal_initializer(stddev=0.01)
        b_init = tf.constant_initializer(value=0.0)

        state_shape = state.get_shape().as_list()
        batch_size = state_shape[0]
        state_length = state_shape[1]
        state_batch = tf.reshape(tensor=state,
                                 shape=[-1, state_shape[2], state_shape[3], state_shape[4]])

        inputs_image = tl.layers.InputLayer(inputs=state_batch, name=name_prefix + 'INPUT_LAYER')
        conv1 = tl.layers.Conv2d(net=inputs_image,
                                 n_filter=self.config.config_dict['CONV1_1_CHANNEL_SIZE'],
                                 filter_size=self.config.config_dict['CONV1_1_FILTER_SIZE'],
                                 strides=self.config.config_dict['CONV1_1_STRIDE_SIZE'],
                                 act=tf.nn.relu,
                                 W_init=W_init,
                                 b_init=b_init,
                                 name=name_prefix + 'CONV1_1_LAYER'
                                 )
        conv1 = tl.layers.Conv2d(net=conv1,
                                 n_filter=self.config.config_dict['CONV1_2_CHANNEL_SIZE'],
                                 filter_size=self.config.config_dict['CONV1_2_FILTER_SIZE'],
                                 strides=self.config.config_dict['CONV1_2_STRIDE_SIZE'],
                                 act=tf.nn.relu,
                                 W_init=W_init,
                                 b_init=b_init,
                                 name=name_prefix + 'CONV1_2_LAYER'
                                 )
        if self.is_training is True:
            conv1 = tl.layers.BatchNormLayer(layer=conv1,
                                             epsilon=0.000001,
                                             is_train=True,
                                             name=name_prefix + 'CONV1_BATCH_NORM_LAYER'
                                             )
        else:
            conv1 = tl.layers.BatchNormLayer(layer=conv1,
                                             epsilon=0.000001,
                                             is_train=False,
                                             name=name_prefix + 'CONV1_BATCH_NORM_LAYER'
                                             )
        pool1 = tl.layers.MaxPool2d(net=conv1,
                                    filter_size=(self.config.config_dict['POOL1_FILTER_SIZE']),
                                    strides=self.config.config_dict['POOL1_STRIDE_SIZE'],
                                    name=name_prefix + 'POOL1_LAYER'
                                    )
        conv2 = tl.layers.Conv2d(net=pool1,
                                 n_filter=self.config.config_dict['CONV2_1_CHANNEL_SIZE'],
                                 filter_size=self.config.config_dict['CONV2_1_FILTER_SIZE'],
                                 strides=self.config.config_dict['CONV2_1_STRIDE_SIZE'],
                                 act=tf.nn.relu,
                                 W_init=W_init,
                                 b_init=b_init,
                                 name=name_prefix + 'CONV2_1_LAYER'
                                 )
        conv2 = tl.layers.Conv2d(net=conv2,
                                 n_filter=self.config.config_dict['CONV2_2_CHANNEL_SIZE'],
                                 filter_size=self.config.config_dict['CONV2_2_FILTER_SIZE'],
                                 strides=self.config.config_dict['CONV2_2_STRIDE_SIZE'],
                                 act=tf.nn.relu,
                                 W_init=W_init,
                                 b_init=b_init,
                                 name=name_prefix + 'CONV2_2_LAYER'
                                 )
        if self.is_training is True:
            conv2 = tl.layers.BatchNormLayer(layer=conv2,
                                             epsilon=0.000001,
                                             is_train=True,
                                             name=name_prefix + 'CONV2_BATCH_NORM_LAYER'
                                             )
        else:
            conv2 = tl.layers.BatchNormLayer(layer=conv2,
                                             epsilon=0.000001,
                                             is_train=True,
                                             name=name_prefix + 'CONV2_BATCH_NORM_LAYER'
                                             )
        pool2 = tl.layers.MaxPool2d(net=conv2,
                                    filter_size=(self.config.config_dict['POOL2_FILTER_SIZE']),
                                    strides=self.config.config_dict['POOL2_STRIDE_SIZE'],
                                    name=name_prefix + 'POOL2_LAYER'
                                    )
        pool2 = tf_contrib_layers.flatten(inputs=pool2.outputs)

        fc1 = tl.layers.InputLayer(inputs=pool2,
                                   name=name_prefix + 'LSTM_FC1_INPUT_LAYER')
        fc1 = tl.layers.DenseLayer(layer=fc1,
                                   n_units=self.config.config_dict['DENSE_LAYER_1_UNIT'],
                                   act=tf.nn.relu,
                                   name=name_prefix + 'DENSE_LAYER_1_LAYER')
        fc2 = tl.layers.DropconnectDenseLayer(layer=fc1,
                                              n_units=self.config.config_dict['DENSE_LAYER_2_UNIT'],
                                              act=tf.nn.sigmoid,
                                              name=name_prefix + 'DENSE_LAYER_2_LAYER',
                                              keep=self.config.config_dict['DROP_OUT_PROB_VALUE'])
        feature_length_per_image = fc2.outputs.get_shape().as_list()[1]
        lstm_input = tf.reshape(fc2.outputs, [-1, state_length, feature_length_per_image])
        # LSTM INPUT IS [BATCH_SIZE, LENGTH, FEATURE_DIM]

        # TODO
        # be aware of the init_state when train a lstm
        lstm_input = tl.layers.InputLayer(inputs=lstm_input, name=name_prefix + 'LSTM_INPUT_LAYER')

        init_state = tf.placeholder(dtype=tf.float32,
                                    shape=[self.config.config_dict['LSTM_LAYERS_NUM'], 2, batch_size,
                                           self.config.config_dict['LSMT_INPUT_LENGTH']])
        state_per_layers = tf.unstack(init_state, axis=0)

        rnn_tuple_state = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(state_per_layers[idx][0], state_per_layers[idx][1])
             for idx in range(self.config.config_dict['LSTM_LAYERS_NUM'])]
        )

        rnn = tl.layers.DynamicRNNLayer(layer=lstm_input,
                                        cell_fn=tf.nn.rnn_cell.LSTMCell,
                                        sequence_length=None,
                                        n_hidden=self.config.config_dict['LSMT_INPUT_LENGTH'],
                                        initial_state=rnn_tuple_state,
                                        n_layer=self.config.config_dict['LSTM_LAYERS_NUM'],
                                        return_last=True,
                                        name=name_prefix + 'LSTM_LAYER'
                                        )

        lstm_fc1 = tl.layers.InputLayer(inputs=rnn.outputs,
                                        name=name_prefix + 'LSTM_FC_INPUT_LAYERS')
        lstm_fc1 = tl.layers.DenseLayer(layer=lstm_fc1,
                                        n_units=self.config.config_dict['LSTM_DENSE_LAYER1_UNIT'],
                                        act=tf.nn.relu,
                                        name=name_prefix + 'LSTM_DENSE_LAYER_1')
        lstm_fc2 = tl.layers.DenseLayer(layer=lstm_fc1,
                                        n_units=self.config.config_dict['LSTM_DENSE_LAYER_2_UNIT'],
                                        act=tf.nn.relu,
                                        name=name_prefix + 'LSTM_DENSE_LAYER_2')

        merged_input = tf.concat(values=[action_net.outputs, lstm_fc2.outputs], axis=1)

        net = tl.layers.InputLayer(inputs=merged_input,
                                   name=name_prefix + 'MERGED_STATE_ACTION_INPUT_LAYER')
        net = tl.layers.DenseLayer(layer=net,
                                   n_units=self.config.config_dict['MERGED_LAYER_1_UNIT'],
                                   act=tf.nn.relu,
                                   name=name_prefix + 'MERGED_DENSE_LAYER_1')
        net = tl.layers.DenseLayer(layer=net,
                                   n_units=self.config.config_dict['MERGED_LAYER_2_UNIT'],
                                   act=tf.nn.relu,
                                   name=name_prefix + 'MERGED_DENSE_LAYER_2')
        return net
