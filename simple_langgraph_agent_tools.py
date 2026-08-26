from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os
from langchain_openai import ChatOpenAI
from typing import Annotated,TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from tavily import TavilyClient
from langgraph.prebuilt import ToolNode,tools_condition

#load env var
load_dotenv()

openai_key= os.getenv("OPENAI_API_KEY")
tavily =os.getenv("TRAVILY_API_KEY")
llm_name="gpt-5-nano"
tavily_client =TavilyClient(api_key=tavily)
client = OpenAI(api_key=openai_key)
model =ChatOpenAI(api_key=openai_key,model=llm_name)
#step 1: Build a basic chatbot
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages:Annotated[list,add_messages]

# Define Tavily search tool with @tool decorator
@tool
def tavily_search(query: str, max_results: int = 2):
    """Search the web using Tavily"""
    return tavily_client.search(query, max_results=max_results)

# Bind tools properly
model_with_tools = model.bind_tools([tavily_search])


def bot(state:State):
    print(state["messages"])
    return {"messages":[model_with_tools.invoke(state["messages"])]}


graph_builder = StateGraph(State)
#Instantiante the basic node with tool node
tool_node =ToolNode(tools=[tool])
graph_builder.add_node("tools",tool_node)

graph_builder.add_node("bot",bot)

graph_builder.set_entry_point("bot")

#graph_builder.set_finish_point("bot")

graph = graph_builder.compile()

graph_builder.add_conditional_edges(
    "bot",
    tools_condition,
)

#res =graph.invoke({"messages":["Hello, how are you?"]})
#print(res["messages"])

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit","exit","q"]:
        print("Goodbye...")
        break
    for event in graph.stream({"messages":("user",user_input)}):
        for value in event.values():
            print("Assistant:",value["messages"][-1])


