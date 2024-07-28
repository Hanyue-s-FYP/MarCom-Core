import copy
from typing import Self, Callable
from pydantic import BaseModel, Field
from langchain_community.llms import Ollama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from db import AgentInfo, AgentMemory
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
    llm = Ollama(model="llama3.1", format="json")
    parser = JsonOutputParser(pydantic_object=AgentAttributeRewrite)
    kvInStr = f"[{",".join([attr.string() for attr in attrs])}]"
    prompt = PromptTemplate(
        template="{action}.\n{format_instructions}",
        input_variables=["action"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    action_desc = f"Rewrite {desc} to provide context to an agent named {name} that is in a simulation in a second person point of view, start with 'You are in a simulation with other agents as {name}, a'"
    action = f"Given the following {{key,value}} pairs in a list {kvInStr}, write a paragraph in a second person point of view, try to fit everything in a short paragraph, do not include special characters in the description"
    chain = prompt | llm | parser
    print(f"Rewriting description for agent")
    res_desc = get_chain_response_json(chain, {"action": action_desc}, ["description"])
    print(f"Rewriting attributes for agent")
    invalid_characters = [
        "{",
        "}",
    ]  # seems like if got {} in response it will become some sort of giberrish d, make sure no such character
    res = get_chain_response_json(
        chain,
        {"action": action},
        ["description"],
        additional_check=lambda res: not any(
            invalid_char in res["description"] for invalid_char in invalid_characters
        ),
    )
    return {
        "description": res_desc["description"].strip()
        + " "
        + res["description"].strip()
    }


class AgentAction(BaseModel):
    action: str = Field("the action to take")
    reason: str = Field("the reason for the action taken")
    additional_data_id: int = Field(
        "additional data id to complete the action, for action TALK, write agent_id, for action BUY, write product_id"
    )
    additional_data_content: str = Field(
        "additional data content to complete the action, for action BUY, write name of the product, for action TALK, write the message"
    )


class Agent:
    parser = JsonOutputParser(pydantic_object=AgentAction)
    # to keep consistency, let agent reply product_id:productname
    actions = {
        "BUY": "to buy a product (additional_data needed: 'product_id, product_name')",
        "SKIP": "to skip this cycle without doing anything else (additional_data needed: none)",
        "TALK": "to talk to another agent (additional_data needed: 'agent_id, message')",
    }
    # prompt template would be pretty much the same for all agents (may need to be changed to support switching to other models)
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
        model: str = "llama3.1",  # allow user change, but default to llama3.1
    ) -> None:
        # basic init the class
        self.id = id
        self.name = name
        self.desc = desc
        self.attrs = attrs
        self.simulation_id = simulation_id
        self.model = model

    # actually initialising the agent, creating the llms etc
    # returns True if the agent is being initialized for the first time (no previous record of rewritten descriptions in the db), False otherwise, so Simulation can insert to the SimulationEvent regarding creation of agent
    def init_agent(self) -> bool:
        first_time = True
        print(f"Initialising agent {self.id} for simulation {self.simulation_id}")
        # get combined description (can get from db if any for consistency, if ntg from db generate lo)
        print(
            f"Getting rewritten version of agent {self.id}, combining attributes and general description"
        )
        self.agent_model = AgentInfo.get_or_none(
            AgentInfo.agent_id == self.id, AgentInfo.sim_id == self.simulation_id
        )
        if self.agent_model is None:
            # obtain the rewritten description
            self.sim_desc = get_agent_desc_rewrite(self.name, self.desc, self.attrs)[
                "description"
            ]
            # write rewritten description of the agent to the db
            self.agent_model = AgentInfo.create(
                agent_id=self.id,
                sim_id=self.simulation_id,
                rewritten_desc=self.sim_desc,
            )
        else:
            # already has record in db
            self.sim_desc = self.agent_model.rewritten_desc
            first_time = False
        print(f"Assigning LLM to agent {self.id}")
        # find if the agent has memory stored in db
        memory_query = (
            AgentMemory.select(AgentMemory)
            .join(AgentInfo)
            .where(AgentMemory.agent == self.agent_model)
            .order_by(AgentMemory.time_created)
        )
        for mem in memory_query:
            self.memory.append(mem.content)
        llm = Ollama(
            model=self.model,
            stop=["<|eot_id|>"],  # might need to change this when switch model
            format="json",
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
        # write to db as well
        AgentMemory.create(agent=self.agent_model, content=mem)

    # if using model that are more powerful maybe can include short description of the agent for more context
    def to_prompt_str(self):
        return f"(agent_id:{self.id})"
