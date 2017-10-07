from src.agent.agent import Agent
from src.model.ddpg.ddpgModel import DDPGModel
from src.config.ddpgConfig import DDPGConfig
from configuration import CONFIG_PATH
from src.model.actor.denseActor import DenseActor
from src.model.critic.denseCritic import DenseCritic
from src.model.critic.LSTMCritic import LSTMCritic
from src.model.actor.LSTMActor import LSTMActor
from src.model.utils.utils import load_json
from configuration.standard_key_list import CONFIG_STANDARD_KEY_LIST


class DDPGAgent(Agent):

    standard_key_list = load_json(CONFIG_STANDARD_KEY_LIST + '/DDPGAgentKeyList.json')

    def __init__(self, env, config, model=None):
        super(DDPGAgent, self).__init__(env=env, config=config, model=model)
        if model is None:
            a = DDPGConfig(config_path=CONFIG_PATH + '/testDDPGConfig.json',
                           ddpg_standard_key_list=DDPGModel.standard_key_list,
                           actor_standard_key_list=LSTMActor.standard_key_list,
                           critic_standard_key_list=LSTMCritic.standard_key_list)

            self.model = DDPGModel(config=a,
                                   actor=LSTMActor,
                                   critic=LSTMCritic)

    def play(self):
        for i in range(self.config.config_dict['EPISODE_COUNT']):
            self.state = self.env.reset()
            for j in range(self.config.config_dict['MAX_STEPS']):
                self.action = self.model.noise_action(state=self.state)
                new_state, reward, done = self.env.step(action=self.action)
                self.model.perceive(state=self.state,
                                    action=self.action,
                                    reward=self.reward,
                                    next_state=new_state,
                                    done=done
                                    )
                if done is True:
                    break
