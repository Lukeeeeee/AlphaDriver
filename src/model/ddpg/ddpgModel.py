from src.model.model import Model
import numpy as np
import src.model.utils as utils
from configuration.standard_key_list import CONFIG_STANDARD_KEY_LIST


class DDPGModel(Model):
    standard_key_list = utils.load_json(file_path=CONFIG_STANDARD_KEY_LIST + '/ddpgKeyList.json')

    def __init__(self, config, actor, critic, sess_flag=False, data=None):
        super(DDPGModel, self).__init__(config, sess_flag, data)
        self.actor = actor(config=config.actor_config)
        self.critic = critic(config=config.critic_config)
        self.replay_buffer = utils.ReplayBuffer(buffer_size=self.config.config_dict['BATCH_SIZE'])
        self.noise = utils.OUNoise(action_dimension=self.config.config_dict['ACTION_DIM'])

    def update(self):
        mini_batch = self.replay_buffer.get_batch(self.config.BATCH_SIZE)
        state_batch = np.asarray([data[0] for data in mini_batch])
        action_batch = np.asarray([data[1] for data in mini_batch])
        reward_batch = np.asarray([data[2] for data in mini_batch])
        next_state_batch = np.asarray([data[3] for data in mini_batch])
        done_batch = np.asarray([data[4] for data in mini_batch])

        next_action_batch = self.predict_target_action(state=next_state_batch)
        q_value_batch = self.predict_target_q_value(state=next_state_batch, action=next_action_batch)
        y_batch = []

        for i in range(len(mini_batch)):
            if done_batch[i] is True:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + self.config.config_dict['GAMMA_REWARD'] * q_value_batch[i])
        y_batch = np.resize(y_batch, [self.config.config_dict['BATCH_SIZE'], 1])

        critic_loss = self.critic.update(sess=self.sess, q_label=y_batch, state=state_batch, action=action_batch)

        action_batch_for_update = self.actor.predict(sess=self.sess, state=state_batch)

        q_gradients = self.critic.compute_action_gradients(sess=self.sess,
                                                           state=state_batch,
                                                           action=action_batch_for_update)

        actor_loss = self.actor.update(sess=self.sess, gradients=q_gradients, state=state_batch)

        self.actor.update_target()
        self.critic.update_target()

        return critic_loss, actor_loss

    def action(self, state):
        action = self.actor.action(sess=self.sess, state=state)
        return action

    def noise_action(self, state):
        action = self.action(state=state)
        action = action + self.noise.noise()
        return action

    def perceive(self, state, action, reward, next_state, done):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

        # Store transitions to replay start size then start training
        if self.replay_buffer.count() > self.config.config_dict['REPLAY_START_SIZE']:
            self.update()
            # if self.time_step % 10000 == 0:
            # self.actor_network.save_network(self.time_step)
            # self.critic_network.save_network(self.time_step)

        # Re-iniitialize the random process when an episode ends
        if done is True:
            self.noise.reset()

    def predict_q_value(self, state, action):
        return self.critic.predict(sess=self.sess,
                                   state=state,
                                   action=action)

    def predict_target_q_value(self, state, action):
        return self.critic.predict_target(sess=self.sess,
                                          state=state,
                                          action=action)

    def predict_action(self, state):
        return self.actor.predict(sess=self.sess,
                                  state=state)

    def predict_target_action(self, state):
        return self.actor.predict_target(sess=self.sess,
                                         state=state)


if __name__ == '__main__':
    from src.config.ddpgConfig import DDPGConfig
    from configuration import CONFIG_PATH
    from src.model.actor.LSTMActor import LSTMActor
    from src.model.critic.LSTMCritic import LSTMCritic

    a = DDPGConfig(config_path=CONFIG_PATH + '/testDDPGConfig.json',
                   ddpg_standard_key_list=DDPGModel.standard_key_list,
                   actor_standard_key_list=LSTMActor.standard_key_list,
                   critic_standard_key_list=LSTMCritic.standard_key_list
                   )
    ddpg = DDPGModel(config=a, actor=LSTMActor, critic=LSTMCritic)
