import copy
from typing import Self, Callable
from pydantic import BaseModel, Field
from langchain_community.llms import Ollama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from db import AgentInfo, AgentMemory
from product import Product
from utils import get_chain_response_json, get_format_instruction_of_pydantic_object


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
    llm = Ollama(model="llama3.1", format="json", stop=["<|eot_id|>"])
    parser = JsonOutputParser(pydantic_object=AgentAttributeRewrite)
    kvInStr = f"[{",".join([attr.string() for attr in attrs])}]"
    prompt = PromptTemplate(
        template="""
            <|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            You are a simulation manager of a simulator that simulates consumer behavior with LLM agents. You are tasked to give the agents about their roles by performing the action.
            {format_instructions}
            <|eot_id|>
            Action: {action}
        """,
        input_variables=["action"],
        partial_variables={
            "format_instructions": "Respond with a JSON object with a single field 'description' where it's value is the output of the performed action."
        },
    )

    rewrite_desc = f"Rewrite {desc} to create an agent named {name} that is in a simulation in a second person point of view, start with 'You are in a simulation with other agents and you act as {name},'"
    rewrite_attr_second_person = f"Given the following {{key,value}} pairs in a list {kvInStr}, write a paragraph in a second person point of view, try to fit everything in a short paragraph, do not include special characters in the description"
    chain = prompt | llm | parser
    invalid_characters = [
        "{",
        "}",
    ]  # seems like if got {} in response it will become some sort of giberrish d, make sure no such character

    def add_check(res):
        if type(res["description"]) != str:
            return False
        return not any(
            invalid_char in res["description"] for invalid_char in invalid_characters
        )

    # rewriting description for agent
    res_desc = get_chain_response_json(
        chain, {"action": rewrite_desc}, ["description"], additional_check=add_check
    )
    # rewriting attributes for agent
    res = get_chain_response_json(
        chain,
        {"action": rewrite_attr_second_person},
        ["description"],
        additional_check=add_check,
    )
    rewrite_second_person = (
        res_desc["description"].strip() + " " + res["description"].strip()
    )

    # rewrite again in 3rd person point of view so can provide to feedback agent
    rewrite_third_person_prompt = (
        f"Rewrite '{rewrite_second_person}' in a third person point of view"
    )
    # rewriting description for agent in third person view
    res_3rd = get_chain_response_json(
        chain,
        {"action": rewrite_third_person_prompt},
        ["description"],
        additional_check=add_check,
    )

    return {
        "description": rewrite_second_person,
        "description_3rd": res_3rd,
    }


class AgentAction(BaseModel):
    action: str = Field("", description="the action to take")
    reason: str = Field("", description="the reason for the action taken")
    additional_data_id: int = Field(
        "", 
        description="additional data id to complete the action, for action MESSAGE, write the id of the agent (eg. 1), for action BUY, write the id of the product(eg. 1), for SKIP, write 0 as the value for this key"
    )
    additional_data_content: str = Field(
        "",
        description="additional data content to complete the action, for action BUY, write name of the product, for action MESSAGE, write the message, for SKIP, write the reason as the value for this key"
    )


# model seems very suka response with only message and not action etc, just craft one class since agent id wont be used anyways
class MessageResponse(BaseModel):
    message: str = Field("", description="the message to send back")


class Agent:
    parser = JsonOutputParser(pydantic_object=AgentAction)
    # to keep consistency, let agent reply product_id:productname
    actions = {
        "BUY": "to buy a product (additional_data needed: 'product_id, product_name')",
        "SKIP": "to skip this cycle without doing anything else (additional_data needed: 'id, reason')",
        "MESSAGE": "to send a message to another agent (additional_data needed: 'agent_id, message')",
    }
    # prompt template would be pretty much the same for all agents (may need to be changed to support switching to other models)
    prompt = PromptTemplate(
        template="""
            <|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            {system_prompt}
            {format_instructions}
            <|eot_id|>
            Valid agents:{agents}
            Valid products:{products}
            Valid actions:{actions}
            Please only take actions in the valid actions list, buy only products in the valid products list, message only agents in the valid agent list.
            {memory}
        """,
        input_variables=["system_prompt", "memory", "agents", "products", "actions"],
        partial_variables={
            "format_instructions": get_format_instruction_of_pydantic_object(
                AgentAction
            ),
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
        # initialising agent for simulation
        # get combined description (can get from db if any for consistency, if ntg from db generate lo)
        self.agent_model = AgentInfo.get_or_none(
            AgentInfo.agent_id == self.id, AgentInfo.sim_id == self.simulation_id
        )
        if self.agent_model is None:
            # obtain the rewritten description
            desc = get_agent_desc_rewrite(self.name, self.desc, self.attrs)
            self.sim_desc = desc["description"]
            self.sim_desc_3rd = desc["description_3rd"]
            # write rewritten description of the agent to the db
            self.agent_model = AgentInfo.create(
                agent_id=self.id,
                sim_id=self.simulation_id,
                rewritten_desc=self.sim_desc,
                rewritten_desc_third_person=self.sim_desc_3rd,
            )
        else:
            # already has record in db
            self.sim_desc = self.agent_model.rewritten_desc
            self.sim_desc_3rd = self.agent_model.rewritten_desc_third_person
            first_time = False
        # find if the agent has memory stored in db
        memory_query = (
            AgentMemory.select(AgentMemory)
            .join(AgentInfo)
            .where(AgentMemory.agent == self.agent_model)
            .order_by(AgentMemory.time_created)
        )
        for mem in memory_query:
            self.add_to_memory(mem.content, save_to_db=False) # alrd in db d the memory
        llm = Ollama(
            model=self.model,
            stop=["<|eot_id|>"],  # might need to change this when switch model
            format="json",
        )
        self.chain = self.prompt | llm | self.parser
        return first_time

    # calls the agent to take action for the cycle
    def get_action(
        self,
        env_desc: str,
        message: str,
        products: list[Product],
        agents: list[Self],
        actions: dict[str, str] = None,
        should_add_memory: bool = False,
    ):
        if should_add_memory:
            self.add_to_memory(message)
        actions = actions if actions is not None else self.actions
        def action_check(res):
            # check if all is of str type (additional_id can be string or int)
            for k in res:
                if res["action"] is str and res["action"] == "SKIP":
                    break
                if k == "additional_data_id" and not (type(res[k]) is str or type(res[k]) is int):
                    return False
                elif k != "additional_data_id" and type(res[k]) is not str:
                    return False
            # noneed care about case (LLM output is very hard to control)
            return res["action"].upper() in actions and res["reason"] != ""
        action = get_chain_response_json(
            self.chain,
            {
                "system_prompt": f"{env_desc}\n{self.sim_desc}",
                "memory": "\n".join(self.memory) 
                + (
                    f"\n{message}" if not should_add_memory else ""
                ),  # if no add to memory, just append as part of the prompt
                "agents": f"[{','.join([agent.to_prompt_str() for agent in agents if int(agent.id) != int(self.id)])}]",
                "products": f"[{';'.join([product.to_prompt_str() for product in products])}]",
                "actions": f"[{';'.join([f'{k}:{v}' for k, v in (actions.items())])}]",
            },
            expected_fields=[k for k in (AgentAction.model_json_schema()["properties"])],
            additional_check=action_check,
        )
        return action

    # when other agent talks to this agent, only can talk back to the agent
    # always add to memory in this case
    def get_talk_response(
        self, env_desc: str, message: str, products: list[Product], agents: list[Self]
    ):
        self.add_to_memory(message)
        # create a specialised chain only for responding back to the message
        llm = Ollama(
            model=self.model,
            stop=["<|eot_id|>"],  # might need to change this when switch model
            format="json",
        )
        parser = JsonOutputParser(pydantic_object=MessageResponse)
        prompt = PromptTemplate(
            template="""
                <|begin_of_text|>
                <|start_header_id|>system<|end_header_id|>
                {system_prompt}
                {format_instructions}
                <|eot_id|>
                {memory}
            """,
            input_variables=["system_prompt", "memory"],
            partial_variables={
                "format_instructions": get_format_instruction_of_pydantic_object(
                    MessageResponse
                ),
            },
        )
        chain = prompt | llm | parser
        return get_chain_response_json(
            chain,
            {
                "system_prompt": f"{env_desc}\n{self.sim_desc}",
                "memory": "\n".join(self.memory),
            },
            expected_fields=["message"],
        )

    # add to agent's memory
    def add_to_memory(self, mem: str, save_to_db: bool = True):
        self.memory.append(mem)
        self.memory = self.memory[-30:] # sliding window (context too less, so only take last 30 otherwise system prompt might get overwritten)
        # write to db as well (if is not called when init agent)
        if save_to_db:
            AgentMemory.create(agent=self.agent_model, content=mem)

    # if using model that are more powerful maybe can include short description of the agent for more context
    def to_prompt_str(self):
        return f"(agent_id:{self.id})"
