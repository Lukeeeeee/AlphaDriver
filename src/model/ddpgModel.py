from src.model.model import Model
import src.model.utils as utils
import numpy as np


class DDPGModel(Model):
    def __init__(self, config, actor, critic, sess_flag=False, data=None):
        super(DDPGModel, self).__init__(config, sess_flag, data)
        self.actor = actor(config=config.actor_config)
        self.critic = critic(config=config.critic_config)
        self.replay_buffer = utils.ReplayBuffer()
        self.noise = utils.OUNoise()

    def update(self):
        mini_batch = self.replay_buffer.get_batch(self.config.BATCH_SIZE)
        state_batch = np.asarray([data[0] for data in mini_batch])
        action_batch = np.asarray([data[1] for data in mini_batch])
        reward_batch = np.asarray([data[2] for data in mini_batch])
        next_state_batch = np.asarray([data[3] for data in mini_batch])
        done_batch = np.asarray([data[4] for data in mini_batch])

        next_action_batch = self.actor.target_predict()
