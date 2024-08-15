import random
from agent import Agent
from db import AgentInfo, SimulationEvent
from product import Product
from pydantic import BaseModel, Field
from langchain_community.llms import Ollama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from utils import get_chain_response_json, get_format_instruction_of_pydantic_object

class Simulation:
    # total_cycle is negative means should run infinitely
    def __init__(
        self,
        id: int,
        env_desc: str,
        agents: list[Agent],
        products: list[Product],
        total_cycle: int = -1,
    ) -> None:
        self.id = id
        self.env_desc = env_desc
        self.agents = agents
        self.products = products
        self.total_cycle = total_cycle
        self.cycle = 1
        self.inited = False
        self.paused = False

    def init_simulation(self):
        # actually initialising the agents
        for a in self.agents:
            first_time = a.init_agent()
            if first_time:
                agent = AgentInfo.create(
                    agent_id=a.id,
                    sim_id=self.id,
                    rewritten_desc=a.sim_desc,
                    rewritten_desc_third_person=a.sim_desc_3rd,
                )
                yield SimulationEvent.create(
                    agent=agent,
                    sim_id=self.id,
                    type="SIMULATION",
                    content=f"Initialised Agent {a.name} with rewritten description: {a.sim_desc}",
                    cycle=self.cycle,
                )
        self.inited = True

    def pause_simulation(self):
        self.paused = True

    def resume_simulation(self):
        self.paused = False

    # use another LLM and generate events for feedbacks on "BUY" and "SKIP" actions
    # also provide the product details and maybe the agent description so the LLM can get more context
    def simulation_response_helper(
        self, action: str, reason: str, env_desc: str, product: Product, agent: Agent
    ) -> str:
        should_positive = random.randrange(1, 10) > 6 # 5050 chance to be positive
        prompt_template = """
            <|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            You are managing a simulation where LLM agents are being used to simulate consumer behavior where they can either buy or skip a certain product. Generate a random event in first person point of view that is {should_positive} relative to the environment description and provide a single short sentence on how effective the product is to the agent relative to the agent description based on the product purchase details, the random event and the environment description. If action is SKIP, generate random events that are not related to the product but relates to the environment description and the agent's aim. {should_positive_enforcement}
            Response format:{format_instructions}
            <|eot_id|>
            Environment Description:{env_desc}
            Purchased Product Details:{product_desc}
            Agent Description:{agent_desc}
            Action:{agent_action}
            Reason:{action_reason}
        """
        llm = Ollama(model="llama3.1", format="json")
        parser = JsonOutputParser(pydantic_object=SimulationActionResp)
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=[
                "env_desc",
                "product_desc",
                "agent_desc",
                "agent_action",
                "action_reason",
            ],
            partial_variables={
                "format_instructions": get_format_instruction_of_pydantic_object(SimulationActionResp),
                "should_positive": "positive" if should_positive else "negative",
                "should_positive_enforcement": "The event should clearly benefit the agent." if should_positive else "The event should clearly harm the agent without any potential for positive interpretation."
            },
        )
        chain = prompt | llm | parser
        return get_chain_response_json(
            chain,
            {
                "env_desc": env_desc,
                "product_desc": product.desc if product is not None else "Agent did not buy any product",
                "agent_desc": agent.sim_desc_3rd,
                "agent_action": action,
                "action_reason": reason,
            },
            ["feedback"],
        )

    # progresses the cycle
    def proceed_cycle(self):
        for agent in self.agents:
            agent.add_to_memory(f"Cycle {self.cycle} start")
        for agent in self.agents:
            prompt_message = f"What action would you like to perform?"
            # obtaining action from agent
            action = agent.get_action(
                self.env_desc,
                prompt_message,
                self.products,
                self.agents,
            )
            agent_model = AgentInfo.select().where(AgentInfo.agent_id == agent.id)
            # talk can go for very long
            while True:
                match action["action"]:
                    case "BUY":
                        # obtain the product purchased by the agent (check if is valid as well)
                        product_to_buy = None
                        prompt_message = ""
                        data_bundle = action["additional_data_id"]
                        # try give bigger tolerance, LLM output hard to control
                        # tolerate two cases: agent_id:id or id
                        if str(data_bundle).isdigit():
                            product_to_buy = [
                                p for p in self.products if int(p.id) == int(data_bundle)
                            ]
                            if len(product_to_buy) != 1:
                                prompt_message = f"product do not exist in environment, valid IDs are [{','.join([str(p.id) for p in self.products])}]"  # id should be unique
                        else:
                            split = data_bundle.split(":")
                            if len(split) > 2:
                                prompt_message = "invalid buy additional data format, please only provide product id or product_id:id"
                            if len(split) == 1 and split[0].isdigit():
                                product_to_buy = [
                                    p for p in self.products if int(p.id) == int(split[0])
                                ]
                                if len(product_to_buy) != 1:
                                    prompt_message = f"product do not exist in environment, valid IDs are [{','.join([str(p.id) for p in self.products])}]"  # id should be unique
                            elif len(split) == 2 and not split[1].isdigit() or len(split) != 2:
                                prompt_message = "invalid buy additional data format, please only provide product id or product_id:id"
                            elif len(split) == 2 and split[1].isdigit():
                                product_to_buy = [
                                    p for p in self.products if int(p.id) == int(split[1])
                                ]
                                if len(product_to_buy) != 1:
                                    prompt_message = f"product do not exist in environment, valid IDs are [{','.join([str(p.id) for p in self.products])}]"  # id should be unique
                        if (
                            product_to_buy is None
                            or len(product_to_buy) != 1
                        ):
                            prompt_message = f"Attempted to buy product with id {data_bundle}, but {prompt_message}"
                            print("Obtained invalid action, retrying:", prompt_message)
                            action = agent.get_action(
                                self.env_desc,
                                prompt_message,
                                self.products,
                                self.agents,
                            )
                        else:
                            # add BUY action to memory and also db
                            agent.add_to_memory(
                                f"You bought Product {product_to_buy[0].name} with reason \"{action['reason']}\""
                            )
                            # create the BUY event and yield it out to facilitate returning to backend
                            event = SimulationEvent.create(
                                agent=agent_model,
                                sim_id=self.id,
                                type="BUY",
                                content=f"{product_to_buy[0].id}:{action['reason']}",
                                cycle=self.cycle,
                            )
                            yield event
                            # generate feedback
                            feedback = self.simulation_response_helper(
                                action="BUY",
                                reason=action["additional_data_content"],
                                env_desc=self.env_desc,
                                product=product_to_buy[0],
                                agent=agent,
                            )
                            agent.add_to_memory(feedback["feedback"])
                            feedback_event = SimulationEvent.create(
                                agent=agent_model,
                                sim_id=self.id,
                                type="ACTION_RESP",
                                content=feedback["feedback"],
                                cycle=self.cycle,
                            )
                            yield feedback_event
                            break
                    case "SKIP":
                        # add SKIP action to memory and also db
                        agent.add_to_memory(f"You did not buy anything with reason \"{action['reason']}\"")
                        # create the SKIP event and yield it out to facilitate returning to backend
                        event = SimulationEvent.create(
                            agent=agent_model,
                            sim_id=self.id,
                            type="SKIP",
                            content="",
                            cycle=self.cycle,
                        )
                        yield event
                        # generate feedback
                        feedback = self.simulation_response_helper(
                            action="SKIP",
                            reason=action["reason"],
                            env_desc=self.env_desc,
                            product=None,
                            agent=agent,
                        )
                        agent.add_to_memory(feedback["feedback"])
                        feedback_event = SimulationEvent.create(
                            agent=agent_model,
                            sim_id=self.id,
                            type="ACTION_RESP",
                            content=feedback["feedback"],
                            cycle=self.cycle,
                        )
                        yield feedback_event
                        break
                    case "MESSAGE":
                        prompt_message = ""
                        agent_to_talk = None
                        # check if recepient exist
                        data_bundle = action["additional_data_id"]
                        # try give bigger tolerance, LLM output hard to control
                        # tolerate two cases: agent_id:id or id
                        if str(data_bundle).isdigit():
                            agent_to_talk = [
                                a for a in self.agents if int(a.id) == int(data_bundle)
                            ]
                            if len(agent_to_talk) != 1:
                                prompt_message = f"agent do not exist in environment, valid IDs are [{','.join([str(a.id) for a in self.agents if int(a.id) != agent.id])}]"  # id should be unique
                        else:
                            split = data_bundle.split(":")
                            if len(split) > 2:
                                prompt_message = "invalid message additional data format, please provide only the agent id or agent_id:id"
                            if len(split) == 1 and split[0].isdigit():
                                agent_to_talk = [
                                    a for a in self.agents if int(a.id) == int(split[0])
                                ]
                                if len(agent_to_talk) != 1:
                                    prompt_message = f"agent do not exist in environment, , valid IDs are [{','.join([str(a.id) for a in self.agents if int(a.id) != agent.id])}]"  # id should be unique
                            elif len(split) == 2 and not split[1].isdigit() or len(split) != 2:
                                prompt_message = "invalid talk additional data format, please provide only the agent id or agent_id:id"
                            elif len(split) == 2 and split[1].isdigit():
                                agent_to_talk = [
                                    a for a in self.agents if int(a.id) == int(split[1])
                                ]
                                if len(agent_to_talk) != 1:
                                    prompt_message = f"agent do not exist in environment, , valid IDs are [{','.join([str(a.id) for a in self.agents if int(a.id) != agent.id])}]"  # id should be unique

                        if (len(agent_to_talk) == 1 and int(agent_to_talk[0].id) == int(agent.id)):
                            prompt_message = f"you cannot message yourself, valid IDs are [{','.join([str(a.id) for a in self.agents if int(a.id) != agent.id])}]"

                        if agent_to_talk is None or len(agent_to_talk) != 1:
                            prompt_message = f"Attempted to message agent with id {data_bundle}, but {prompt_message}"
                            print("Obtained invalid action, retrying:", prompt_message)
                            action = agent.get_action(
                                self.env_desc,
                                prompt_message,
                                self.products,
                                self.agents,
                            )
                        else:
                            # if prompt message has not been set then should be no error alrd
                            if prompt_message == "":
                                prompt_message = f"Agent {agent.id} sends you a message:{action['additional_data_content']}, what would you like to reply?"
                            # add to memory of the sending agent so it is aware that it sent a message to another agent
                            agent.add_to_memory(
                                f"You sent agent {agent_to_talk[0].id} a message: {action['additional_data_content']}"
                            )
                            # create the MESSAGE event and yield it out to facilitate returning to backend
                            event = SimulationEvent.create(
                                agent=agent_model,
                                sim_id=self.id,
                                type="MESSAGE",
                                content=f"{agent_to_talk[0].id}:{action['additional_data_content']}",
                                cycle=self.cycle,
                            )
                            yield event
                            # agent_to_talk model
                            agent_model_next = AgentInfo.select().where(AgentInfo.agent_id == agent_to_talk[0].id)
                            action_next = agent_to_talk[0].get_talk_response(
                                self.env_desc, prompt_message, self.products, [agent]
                            )  # message obtained from other agent, reforward to this agent and can rerun this big while loop
                            reply_event = SimulationEvent.create(
                                agent=agent_model_next,
                                sim_id=self.id,
                                type="MESSAGE",
                                content=f"{agent.id}:{action_next['message']}",
                                cycle=self.cycle,
                            )
                            yield reply_event
                            # can no need care if it's return to this agent d, just forward back
                            action = agent.get_action(
                                self.env_desc,
                                f"Agent {agent_to_talk[0].id} replies you:{action_next['message']}",
                                self.products,
                                self.agents,
                                should_add_memory=True,
                            )
        self.cycle += 1

    def run_simulation(self):
        if not self.inited:
            for simulation_init_event in self.init_simulation():
                yield simulation_init_event
        while self.cycle <= self.total_cycle:
            for event in self.proceed_cycle():
                yield event


# helper object to get structured response
class SimulationActionResp(BaseModel):
    feedback: str = Field(
        description="the feedback to be given to the agent that performed the action"
    )
