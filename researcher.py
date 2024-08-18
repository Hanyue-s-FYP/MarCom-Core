# Researcher is a llm agent that browses duckduckgo to research for potential competitors for a product
from typing import Any
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from product import Product
from utils import get_chain_response_json

# default to llama3.1 now, dont have the ability to switch models yet becuz llama need structured prompt guidelines and the guidelines aren't exactly usable with other models
llm = ChatOllama(
    model="llama3.1", format="json", temperature=0
)  # set temp to 0 so the model dont do anything too creative :)) which is bad when doing some serious researching tasks

# Web Search Tool initialisation, using duckduckgo cuz it's free and doesn't require too much setup
wrapper = DuckDuckGoSearchAPIWrapper(max_results=25)
web_search_tool = DuckDuckGoSearchResults(api_wrapper=wrapper)


def do_web_search(query: str) -> list[dict[str, str]]:
    print("Searching on DuckDuckGo")
    return web_search_tool.invoke(query)


# transform the query given to smtg more optimised for searches first (will provide product name, description and price)
query_prompt = PromptTemplate(
    template="""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|> 
    You are an expert at crafting web search queries for market competition research questions.
    More often than not, a user will provide information (which may include name, description and price) about a product that may or may not exist in the market and wants to find similar products that may be of competition to that product, however it might not be suitable to be used directly as a search query. 
    Construct a query to be the most effective web search string possible based on their product information.
    Return the JSON with a single key 'query' with no premable or explanation. 
    Product information: 
    Product Name: {name}
    Product Description: {desc}
    Product Price: RM{price} 
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["name", "desc", "price"],
)

query_chain = (
    query_prompt | llm | JsonOutputParser()
)  # such simple json shouldnt need to much guidance gua


def reconstruct_query_with_product(p: Product):
    print("Reconstructing query from product")
    # but still, check if it follows the format
    return get_chain_response_json(
        query_chain,
        {
            "name": p.name,
            "desc": p.desc,
            "price": "{:.2f}".format(p.price),
        },
        expected_fields=["query"],
    )


# generate report based on web search results (for report no need JSON, directly take the text response can d)
report_prompt = PromptTemplate(
    template="""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|> 
    You are an AI assistant for Market Competition Research Question Tasks. You are good at determining which product might be a competition to the user's product. 
    A user will provide information (which may include name, description and price) about a product that may or may not exist in the market and wants to find similar products that may be of competition to that product.
    Strictly use the following pieces of web search context to determine if which products would be able to compete with the user's product and why so. If you don't know the answer, just say that you don't know. 
    Keep the answer concise, but provide all of the details you can in the form of a research report. 
    Only make direct references to material provided in the context. Cite the references to the material together with the link in the end of the report. Strip any preambles and dive straight into the possible competitors.
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Product Information: 
    Product Name: {name}
    Product Description: {desc}
    Product Price: RM{price} 
    Original Query: {ori_query}
    Web Search Context: {context} 
    Answer: 
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["name", "desc", "price", "ori_query", "context"],
)
llm_text = ChatOllama(
    model="llama3.1", temperature=0
)  # set temp to 0 so the model dont do anything too creative :)) which is bad when doing some serious researching tasks
report_chain = report_prompt | llm_text | StrOutputParser()


def get_product_comp_report(p: Product, ori_query: str, web_context: Any):
    print("Obtaining competitor report")
    return report_chain.invoke(
        {
            "name": p.name,
            "desc": p.desc,
            "price": "{:.2f}".format(p.price),
            "ori_query": ori_query,
            "context": web_context,
        }
    )
