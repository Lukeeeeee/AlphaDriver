from src.model.preTrainer.preTrainer import PreTrainer
import numpy as np
from src.model.ddpg.ddpgModel import DDPGModel


class LSTMDDPGPreTrainer(PreTrainer):
    def __init__(self, config, data, lstm_ddpg_model, sess_flag=False):
        if type(lstm_ddpg_model) is not DDPGModel:
            raise NotImplementedError('Pre-trainer dose not support this %s model to pre-train'
                                      % str(lstm_ddpg_model))

        super(LSTMDDPGPreTrainer, self).__init__(config, data, lstm_ddpg_model, sess_flag)

    def pre_train(self):
        for i in range(self.config.config_dict['EPOCH']):

            mini_batch = self.data.return_batch_data(self.config.config_dict['BATCH_SIZE'])
            state = mini_batch['STATE_BATCH']
            action = mini_batch['ACTION_BATCH']
            reward = mini_batch['REWARD_BATCH']
            next_state = mini_batch['NEXT_STATE_BATCH']
            done = mini_batch['DONE_BATCH']

            # TODO SAMPLE NEXT ACTION FROM POLICY OR FROM HUMAN?
            next_action = self.pre_trained_model.predict_target_action(state=next_state)
            q_value = self.pre_trained_model.predict_target_q_value(state=next_state,
                                                                    action=next_action)
            y_batch = []
            for i in range(len(mini_batch)):
                if done[i] is True:
                    y_batch.append(reward[i])
                else:
                    y_batch.append(reward[i] + self.pre_trained_model.config.config_dict['GAMMA_REWARD'] * q_value[i])

            y_batch = np.reshape(y_batch, [self.config.config_dict['BATCH_SIZE'], 1])

            critic_loss = self.pre_trained_model.critic.update(sess=self.pre_trained_model.sess,
                                                               q_label=y_batch,
                                                               state=state,
                                                               action=action)
            q_gradients = self.pre_trained_model.critic.compute_action_gradients(sess=self.pre_trained_model.sess,
                                                                                 state=state,
                                                                                 action=action)
            actor_loss = self.pre_trained_model.actor.update(sess=self.pre_trained_model.sess,
                                                             gradients=q_gradients,
                                                             state=state)
