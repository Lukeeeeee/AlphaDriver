from src.model.model import Model
import src.model.utils as utils
import numpy as np
import tensorflow as tf
import tensorlayer as tl


class DDPGModel(Model):
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

        next_action_batch = self.actor.predict_target(sess=self.sess, state=next_state_batch)
        q_value_batch = self.critic.predict_target(sess=self.sess, state=next_state_batch, action=next_action_batch)
        y_batch = []

        for i in range(len(mini_batch)):
            if done_batch[i] is True:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + self.config.config_dict['GAMMA_REWARD'] * q_value_batch[i])
        y_batch = np.resize(y_batch, [self.config.config_dict['BATCH_SIZE'], 1])

        critic_loss = self.critic.update(sess=self.sess, q_label=y_batch, state=state_batch, action=action_batch)

        action_batch_for_update = self.actor.predict(sess=self.sess, state=state_batch)

        q_gradients = self.critic.predict(sess=self.sess, state=state_batch, action=action_batch_for_update)

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

    # def save_model(self, global_step):
            #     tl.files.save_ckpt(sess=self.sess, save_dir=self.config_json['MODEL_SAVE_DIR'], global_step=global_step)
    #
    # def load_model(self, global_step=None):
    #     if global_step:
    #         tl.files.load_ckpt(sess=self.sess,
            #                            save_dir=self.config_json['MODEL_LOAD_DIR'] + '/model.ckpt-' + str(global_step))
    #     else:
    #         tl.files.load_ckpt(sess=self.sess,
            #                            save_dir=self.config_json['MODE_LOAD_DIR'],
    #                            is_latest=True)
    #


if __name__ == '__main__':
    from src.config.ddpgConfig import DDPGConfig
    from configuration import CONFIG_PATH
    from src.model.actor.denseActor import DenseActor
    from src.model.critic.denseCritic import DenseCritic
    from configuration.standard_key_list import ddpgKeyList

    a = DDPGConfig(config_path=CONFIG_PATH + '/testDDPGConfig.json', standard_key_list=ddpgKeyList.key_list)
    ddpg = DDPGModel(config=a, actor=DenseActor, critic=DenseCritic)
