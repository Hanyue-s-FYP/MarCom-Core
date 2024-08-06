import random
from agent import Agent
from db import SimulationEvent
from product import Product
from pydantic import BaseModel, Field
from langchain_community.llms import Ollama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from utils import get_chain_response_json


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

    def init_simulation(self):
        # actually initialising the agents
        for a in self.agents:
            first_time = a.init_agent()
            if first_time:
                yield SimulationEvent.create(
                    sim_id=self.id,
                    type="SIMULATION",
                    content=f"Initialised Agent {a.name} with rewritten description: {a.sim_desc}",
                )

    # use another LLM and generate events for feedbacks on "BUY" and "SKIP" actions
    # also provide the product details and maybe the agent description so the LLM can get more context
    def simulation_response_helper(
        action: str, reason: str, env_desc: str, product: Product, agent: Agent
    ) -> str:
        should_positive = random.choice([True, False])
        prompt_template = """
            <|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            You are managing a simulation where LLM agents are being used to simulate consumer behavior where they can either buy or skip a certain product. Generate a {should_positive} random event relative to the environment description and provide a single short sentence on how effective the product is to the agent relative to the agent description based on the product purchase details, the random event and the environment description.
            {format_instructions}
            <|eot_id|>
            Environment Description: {env_desc}
            Purchased Product Details: {product_desc}
            Agent Description: {agent_desc}
            Action: {agent_action}
            Reason: {action_reason}
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
                "format_instructions": parser.get_format_instructions(),
                "should_positive": "positive" if should_positive else "negative",
            },
        )
        chain = prompt | llm | parser
        return get_chain_response_json(
            chain,
            {
                "env_desc": env_desc,
                "product_desc": product.desc,
                "agent_desc": agent.sim_desc_3rd,
                "agent_action": action,
                "action_reason": reason,
            },
            ["feedback"],
        )

    # progresses the cycle
    def proceed_cycle(self):
        for agent in self.agents:
            prompt_message = f"Cycle {self.cycle} start"
            # obtaining action from agent
            action = agent.get_action(
                self.env_desc,
                prompt_message,
                self.products,
                self.agents,
                should_add_memory=True,
            )
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
                                p for p in self.products if p.id == data_bundle
                            ]
                            if len(product_to_buy) != 1:
                                prompt_message = "Product do not exist in environment"  # id should be unique
                        else:
                            split = data_bundle.split(":")
                            if len(split) > 2:
                                prompt_message = "Invalid buy additional data format, please only provide product id or product_id:id"
                            if len(split) == 1 and split[0].isdigit():
                                product_to_buy = [
                                    p for p in self.products if p.id == int(split[0])
                                ]
                                if len(agent_to_talk) != 1:
                                    prompt_message = "Product do not exist in environment"  # id should be unique
                            elif len(split) == 2 and not split[1].isdigit():
                                prompt_message = "Invalid buy additional data format, please only provide product id or product_id:id"
                            elif len(split) == 2 and split[1].isdigit():
                                product_to_buy = [
                                    p for p in self.products if p.id == int(split[1])
                                ]
                                if len(agent_to_talk) != 1:
                                    prompt_message = "Product do not exist in environment"  # id should be unique
                        if (
                            product_to_buy is None
                            or len(product_to_buy) != 1
                            or not prompt_message
                        ):
                            action = agent.get_action(
                                self.env_desc,
                                prompt_message,
                                self.products,
                                self.agents,
                            )
                        else:
                            # add BUY action to memory and also db
                            agent.add_to_memory(
                                f"You bought Product {product_to_buy[0].name} with reason \"{action['additional_data_content']}\""
                            )
                            # create the BUY event and yield it out to facilitate returning to backend
                            event = SimulationEvent.create(
                                agent=agent,
                                sim_id=self.id,
                                type="BUY",
                                content=f"{product_to_buy[0].id}:{action['additional_data_content']}",
                            )
                            yield event
                            # generate feedback
                            feedback = self.simulation_response_helper(
                                "BUY",
                                action["additional_data_content"],
                                self.env_desc,
                                product_to_buy,
                                agent,
                            )
                            agent.add_to_memory(feedback["feedback"])
                            feedback_event = SimulationEvent.create(
                                agent=agent,
                                sim_id=self.id,
                                type="ACTION_RESP",
                                content=feedback["feedback"],
                            )
                            yield feedback_event
                        break
                    case "SKIP":
                        # add SKIP action to memory and also db
                        agent.add_to_memory("You did not buy anything")
                        # create the SKIP event and yield it out to facilitate returning to backend
                        event = SimulationEvent.create(
                            agent=agent,
                            sim_id=self.id,
                            type="SKIP",
                            content="",
                        )
                        yield event
                        # generate feedback
                        feedback = self.simulation_response_helper(
                            "SKIP",
                            "",
                            self.env_desc,
                            product_to_buy,
                            agent,
                        )
                        agent.add_to_memory(feedback["feedback"])
                        feedback_event = SimulationEvent.create(
                            agent=agent,
                            sim_id=self.id,
                            type="ACTION_RESP",
                            content=feedback["feedback"],
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
                                a for a in self.agents if a.id == data_bundle
                            ]
                            if len(agent_to_talk) != 1:
                                prompt_message = "Agent do not exist in environment"  # id should be unique
                        else:
                            split = data_bundle.split(":")
                            if len(split) > 2:
                                prompt_message = "Invalid talk additional data format, please provide only the agent id or agent_id:id"
                            if len(split) == 1 and split[0].isdigit():
                                agent_to_talk = [
                                    a for a in self.agents if a.id == int(split[0])
                                ]
                                if len(agent_to_talk) != 1:
                                    prompt_message = "Agent do not exist in environment"  # id should be unique
                            elif len(split) == 2 and not split[1].isdigit():
                                prompt_message = "Invalid talk additional data format, please provide only the agent id or agent_id:id"
                            elif len(split) == 2 and split[1].isdigit():
                                agent_to_talk = [
                                    a for a in self.agents if a.id == int(split[1])
                                ]
                                if len(agent_to_talk) != 1:
                                    prompt_message = "Agent do not exist in environment"  # id should be unique

                        if agent_to_talk is None or len(agent_to_talk) == 0:
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
                                agent=agent,
                                sim_id=self.id,
                                type="MESSAGE",
                                content=f"{agent_to_talk[0].id}:{action['additional_data_content']}",
                            )
                            yield event
                            action_next = agent_to_talk[0].get_talk_response(
                                self.env_desc, prompt_message, self.products, [agent]
                            )  # message obtained from other agent, reforward to this agent and can rerun this big while loop
                            reply_event = SimulationEvent.create(
                                agent=agent_to_talk[0],
                                sim_id=self.id,
                                type="MESSAGE",
                                content=f"{agent.id}:{action['additional_data_content']}"
                            )
                            yield reply_event
                            # can no need care if it's return to this agent d, just forward back
                            action = agent.get_action(
                                self.env_desc,
                                f"Agent {agent_to_talk[0].id} replies you:{action_next['additional_data_content']}",
                                self.products,
                                self.agents,
                                should_add_memory=True,
                            )
        self.cycle += 1

    def run_simulation(self):
        while self.cycle <= self.total_cycle:
            for event in self.proceed_cycle():
                yield event

# helper object to get structured response
class SimulationActionResp(BaseModel):
    feedback: str = Field(
        description="the feedback to be given to the agent that performed the action"
    )
