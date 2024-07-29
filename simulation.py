from agent import Agent
from db import AgentInfo, SimulationEvent
from product import Product

class Simulation:
    # total_cycle is negative means should run infinitely
    def __init__(self, id: int, env_desc: str, agents: list[Agent], products: list[Product], total_cycle: int = -1) -> None:
        self.id = id
        self.env_desc = env_desc
        self.agents = agents
        self.products = products
        self.total_cycle = total_cycle
        self.cycle = 1

        # actually initialising the agents
        for a in agents:
            first_time = a.init_agent()
            if first_time:
                SimulationEvent.create(sim_id=self.id, type="SIMULATION", content=f"Initialised Agent {a.name} with rewritten description: {a.sim_desc}")

    # progresses the cycle
    def proceed_cycle(self):
        for agent in self.agents:
            prompt_message = f"Cycle {self.cycle} start"
            print(f"Obtaining action from agent {agent.id}")
            action = agent.get_action(self.env_desc, prompt_message, self.products, self.agents)
            # talk can go for very long
            while True:
                print(action)
                match action["action"]:
                    case "BUY":
                        break
                    case "SKIP":
                        break
                    case "TALK":
                        prompt_message = ""
                        agent_to_talk = None
                        # check if recepient exist
                        data_bundle = action["additional_data_id"]
                        # try give bigger tolerance, LLM output hard to control
                        # tolerate two cases: agent_id:id or id
                        if str(data_bundle).isdigit():
                            agent_to_talk = [a for a in self.agents if a.id == data_bundle]
                            if len(agent_to_talk) != 1:
                                prompt_message = "Agent do not exist in environment" # id should be unique
                        else:
                            split = data_bundle.split(":")
                            if len(split) > 2:
                                prompt_message = "Invalid talk additional data format, please provide only the agenrt id or agent_id:id"
                            if len(split) == 1 and split[0].isdigit():
                                agent_to_talk = [a for a in self.agents if a.id == int(split[0])]
                                if len(agent_to_talk) != 1:
                                    prompt_message = "Agent do not exist in environment" # id should be unique
                            elif len(split) == 2 and not split[1].isdigit():        
                                prompt_message = "Invalid talk additional data format, please provide only the agenrt id or agent_id:id"
                            elif len(split) == 2 and split[1].isdigit():
                                agent_to_talk = [a for a in self.agents if a.id == int(split[1])]
                                if len(agent_to_talk) != 1:
                                    prompt_message = "Agent do not exist in environment" # id should be unique
                        
                        # if prompt message has not been set then should be no error alrd
                        if prompt_message == "":
                            prompt_message = f"Agent {agent.id} sends you a message:{action['additional_data_content']}, what would you like to reply?"

                        if agent_to_talk is None:
                            action = agent.get_action(self.env_desc, prompt_message, self.products, self.agents)
                        else:
                            action_next = agent_to_talk[0].get_talk_response(self.env_desc, prompt_message, self.products, [agent]) # message obtained from other agent, reforward to this agent and can rerun this big while loop
                            print(action_next)
                            # can no need care if it's return to this agent d, just forward back
                            action = agent.get_action(self.env_desc, f"Agent {agent_to_talk[0].id} replies you:{action_next['additional_data_content']}", self.products, self.agents)
        self.cycle += 1
