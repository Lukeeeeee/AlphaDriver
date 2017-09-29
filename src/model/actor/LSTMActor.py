from src.model.actor.actor import Actor
import tensorlayer as tl
import tensorflow as tf
import src.model.utils.utils as utils
from configuration.standard_key_list import CONFIG_STANDARD_KEY_LIST


class LSTMActor(Actor):
    standard_key_list = utils.load_json(file_path=CONFIG_STANDARD_KEY_LIST + '/lstmActorKeyList.json')

    def __init__(self, config, sess_flag=False, data=None):
        super(LSTMActor, self).__init__(config, sess_flag, data)
        self.net = self.create_model(state=self.state, name_prefix='ACTOR_')
        self.target_net = self.create_model(state=self.target_state, name_prefix='TARGET_ACTOR')
        self.optimizer, self.optimize_loss = self.create_training_method()

    def create_model(self, state, name_prefix):
        W_init = tf.truncated_normal_initializer(stddev=0.01)
        b_init = tf.constant_initializer(value=0.0)

        image_state_shape = state('IMAGE').get_shape().as_list()
        batch_size = image_state_shape[0]
        state_length = image_state_shape[1]
        state_batch = tf.reshape(tensor=state('IMAGE'),
                                 shape=[-1, image_state_shape[2], image_state_shape[3], image_state_shape[4]])

        inputs_image = tl.layers.InputLayer(inputs=state_batch, name=name_prefix + 'INPUT_LAYER_' + 'IMAGE')

        merge_tensor_dict = {}
        for name, tensor in state().items():
            if name != 'IMAGE':
                merge_tensor_dict[name] = tensor
        merged_flattened_input = utils.flatten_and_concat_tensors(name_prefix=name_prefix,
                                                                  tensor_dict=merge_tensor_dict)

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
        pool2 = tl.layers.FlattenLayer(layer=pool2,
                                       name=name_prefix + 'POOL2_FLATTEN_LAYER')

        fc1 = tl.layers.DenseLayer(layer=pool2,
                                   n_units=self.config.config_dict['DENSE_LAYER_1_UNIT'],
                                   act=tf.nn.relu,
                                   name=name_prefix + 'DENSE_LAYER_1_LAYER')
        fc2 = tl.layers.DropconnectDenseLayer(layer=fc1,
                                              n_units=self.config.config_dict['DENSE_LAYER_2_UNIT'],
                                              act=tf.nn.sigmoid,
                                              name=name_prefix + 'DENSE_LAYER_2_LAYER',
                                              keep=self.config.config_dict['DROP_OUT_PROB_VALUE'])
        feature_layer = tl.layers.ConcatLayer(layer=[fc2, merged_flattened_input],
                                              concat_dim=1,
                                              name=name_prefix + 'LSTM_FEATURE_CONCAT_LAYER')
        feature_length = feature_layer.outputs.get_shape().as_list()[1]
        # LSTM INPUT IS [BATCH_SIZE, LENGTH, FEATURE_DIM]
        lstm_input = tl.layers.ReshapeLayer(layer=feature_layer,
                                            shape=[-1, state_length, feature_length],
                                            name=name_prefix + 'LSTM_FEATURE_RESHAPE_LAYER')
        # TODO
        # be aware of the init_state when train a lstm

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

        lstm_fc1 = tl.layers.DenseLayer(layer=rnn,
                                        n_units=self.config.config_dict['LSTM_DENSE_LAYER1_UNIT'],
                                        act=tf.nn.relu,
                                        name=name_prefix + 'LSTM_DENSE_LAYER_1')
        lstm_fc2 = tl.layers.DenseLayer(layer=lstm_fc1,
                                        n_units=self.config.config_dict['LSTM_DENSE_LAYER_2_UNIT'],
                                        act=tf.nn.tanh,
                                        name=name_prefix + 'LSTM_DENSE_LAYER_2')
        return lstm_fc2

    def create_training_method(self):
        parameters_gradients = tf.gradients(self.action, self.var_list, -self.q_value_gradients)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.config_dict['LEARNING_RATE'])
        optimize_loss = optimizer.apply_gradients(grads_and_vars=zip(parameters_gradients, self.var_list))
        return optimizer, optimize_loss


if __name__ == '__main__':
    from src.config.config import Config
    from configuration import CONFIG_PATH

    a = Config(config_dict=None, standard_key_list=LSTMActor.standard_key_list)
    a.load_config(path=CONFIG_PATH + '/testLSTMActorConfig.json')
    actor = LSTMActor(config=a)
    with tf.Session() as sess:
        with sess.as_default():
            tl.layers.initialize_global_variables(sess)
            actor.net.print_params()
    pass