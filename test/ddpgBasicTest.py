from src.environment.torcsEnvironment.torcsEnvironment import TorcsEnvironment
from src.model.ddpg.ddpgModel import DDPGModel
from src.agent.ddpgAgent import DDPGAgent
from src.model.utils.utils import load_json
from src.config.ddpgConfig import DDPGConfig
from configuration import CONFIG_PATH
from src.model.actor.LSTMActor import LSTMActor
from src.model.critic.LSTMCritic import LSTMCritic


def create_agent():
    env_config = load_json(file_path=CONFIG_PATH + '/testTorcsEnvironmentConfig.json')
    env = TorcsEnvironment(env_config)
    agent_config = load_json(file_path=CONFIG_PATH + '/testDDPGAgentConfig.json')
    a = DDPGConfig(config_path=CONFIG_PATH + '/testDDPGConfig.json',
                   ddpg_standard_key_list=DDPGModel.standard_key_list,
                   actor_standard_key_list=LSTMActor.standard_key_list,
                   critic_standard_key_list=LSTMCritic.standard_key_list
                   )
    ddpg = DDPGModel(config=a, actor=LSTMActor, critic=LSTMCritic)

    agent = DDPGAgent(env=env, config=agent_config, model=ddpg)


if __name__ == '__main__':
    create_agent()
