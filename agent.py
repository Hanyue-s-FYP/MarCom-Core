import copy
from typing import Self, Callable
from pydantic import BaseModel, Field
from langchain_community.llms import Ollama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from product import Product
from utils import get_chain_response_json


class AgentAttribute:
    def __init__(self, key: str, value: str) -> None:
        self.key = key
        self.value = value

    def string(self) -> str:
        return f"{{{self.key}, {self.value}}}"  # for future me: need to use double curly to escape curly in f strings


# for use by json output parser
class AgentAttributeRewrite(BaseModel):
    description: str = Field(description="rewritten paragraph")


def get_agent_desc_rewrite(
    name: str, desc: str, attrs: list[AgentAttribute]
) -> dict[str, str]:
    llm = Ollama(model="llama3")
    parser = JsonOutputParser(pydantic_object=AgentAttributeRewrite)
    kvInStr = f"[{",".join([attr.string() for attr in attrs])}]"
    prompt = PromptTemplate(
        template="{action}.\n{format_instructions}.",
        input_variables=["action"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    action_desc = f"Given description about the agent {name} in third person view: {desc}, rewrite it so that it provides context about an agent that is in a simulation, hinting there are other agents in the simulation, starting with 'You are in a simulation with other agents as {name}', write it in second person point of view in a short paragraph and be concise"
    action = f"Given the following {{key,value}} pairs in a list {kvInStr}, write a paragraph in a second person point of view, try to fit everything in a short paragraph, be concise"
    chain = prompt | llm | parser
    print(f"Obtaining instruction rewrite on description for agent")
    res_desc = get_chain_response_json(chain, {"action": action_desc}, ["description"])
    print(f"Obtaining instruction rewrite on attributes for agent")
    res = get_chain_response_json(chain, {"action": action}, ["description"])
    return {"description": res_desc["description"] + res["description"]}


class AgentAction(BaseModel):
    action: str = Field("the action to take")
    reason: str = Field("the reason for the action taken")
    additional_data_id: int = Field(
        "additional data id to complete the action, for example, to message agent with id 1, write 1, to buy product 1, write 1"
    )
    additional_data_content: str = Field(
        "additional data content to complete the action, for example, for action BUY, write name of the product, for action TALK, write the message"
    )


class Agent:
    parser = JsonOutputParser(pydantic_object=AgentAction)
    # to keep consistency, let agent reply product_id:productname
    actions = {
        "BUY": "to buy a product (additional_data needed: 'product_id:product_name')",
        "SKIP": "to skip this cycle without doing anything else (additional_data needed: '')",
        "TALK": "to talk to another agent (additional_data needed: 'agent_id:message')",
    }
    # prompt template would be pretty much the same for all agents
    prompt = PromptTemplate(
        template="""
            <|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            {system_prompt}
            <|eot_id|>
            {memory}
            All agents:{agents}
            Available products:{products}
            Available actions:{actions}
            {format_instructions}
        """,
        input_variables=["system_prompt", "memory", "agents", "products", "actions"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
        },
    )

    # memory yinggai will be implemented with sliding window, meaning only newest nth cycle memory would be retained
    memory: list[str] = []

    def __init__(
        self,
        id: int,
        name: str,
        desc: str,
        attrs: list[AgentAttribute],
        simulation_id: int,
    ) -> None:
        print(f"Initialising agent {id} for simulation {simulation_id}")
        # basic init the class
        self.id = id
        self.name = name
        self.desc = desc
        self.attrs = attrs
        self.simulation_id = simulation_id
        # get combined description (can get from db if any for consistency, if ntg from db generate lo)
        print(
            f"Getting rewritten version of agent {self.id}, combining attributes and general description"
        )
        self.sim_desc = get_agent_desc_rewrite(self.name, self.desc, self.attrs)[
            "description"
        ]
        print(f"Assigning LLM to agent {self.id}")
        llm = Ollama(
            model="llama3",
            stop=["<|eot_id|>"],
        )
        print(
            f"Assigning memory to agent {self.id}"
        )  # maybe regain memory from db in future?
        self.chain = self.prompt | llm | self.parser

    # calls the agent to take action for the cycle
    def get_action(
        self,
        env_desc: str,
        message: str,
        products: list[Product],
        agents: list[Self],
        actions: dict[str, str] = None,
    ):
        self.add_to_memory(message)
        actions = actions if actions is not None else self.actions
        action = get_chain_response_json(
            self.chain,
            {
                "system_prompt": f"{env_desc}\n{self.sim_desc}",
                "memory": "\n".join(self.memory),
                "agents": f"[{','.join([agent.to_prompt_str() for agent in agents])}]",
                "products": f"[{','.join([product.to_prompt_str() for product in products])}]",
                "actions": ";".join([f"{k}:{v}" for k, v in (actions.items())]),
            },
            expected_fields=[k for k, _ in AgentAction().__dict__.items()],
            additional_check=lambda res: (
                res["action"].upper() in actions
            ),  # noneed care about case (LLM output is very hard to control)
        )
        return action

    # when other agent talks to this agent, only can talk back to the agent
    def get_talk_response(
        self, env_desc: str, message: str, products: list[Product], agents: list[Self]
    ):
        available_actions = {"TALK": self.actions["TALK"]}
        return self.get_action(env_desc, message, products, agents, available_actions)

    # add to agent's memory
    def add_to_memory(self, mem: str):
        self.memory.append(mem)

    def to_prompt_str(self):
        return f"(agent_id:{self.id})"
