from src.agent.agent import Agent
from src.model.ddpg.ddpgModel import DDPGModel
from src.config.ddpgConfig import DDPGConfig
from configuration import CONFIG_PATH
from src.config.utils import load_json
from src.model.actor.denseActor import DenseActor
from src.model.critic.denseCritic import DenseCritic
from configuration.standard_key_list import CONFIG_STANDARD_KEY_LIST


class DDPGAgent(Agent):
    def __init__(self, env, config, model=None):
        super(DDPGAgent, self).__init__(env=env, config=config, model=model)
        if model is None:
            key_list = load_json(file_path=CONFIG_STANDARD_KEY_LIST + '/ddpgKeyList.json')
            a = DDPGConfig(config_path=CONFIG_PATH + '/testDDPGConfig.json', ddpg_standard_key_list=key_list)

            self.model = DDPGModel(config=a,
                                   actor=DenseActor,
                                   critic=DenseCritic)

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
