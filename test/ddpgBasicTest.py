from src.environment.torcsEnvironment.torcsEnvironment import TorcsEnvironment
from src.model.ddpg.ddpgModel import DDPGModel
from src.agent.ddpgAgent import DDPGAgent

def create_agent():
    env = TorcsEnvironment()
    agent = DDPGAgent()