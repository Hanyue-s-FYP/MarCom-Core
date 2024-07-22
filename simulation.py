from agent import Agent
from product import Product


class Simulation:
    # total_cycle is negative means should run infinitely
    def __init__(self, env_desc: str, agents: list[Agent], products: list[Product], total_cycle: int = -1) -> None:
        self.env_desc = env_desc
        self.agents = agents
        self.products = products
        self.total_cycle = total_cycle
        self.cycle = 1

    # progresses the cycle
    def proceed_cycle(self):
        for agent in self.agents:
            print(f"Obtaining action from agent {agent.id}")
            action = agent.get_action(self.env_desc, self.products, self.agents, self.cycle)
            print(action)
