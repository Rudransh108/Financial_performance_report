from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os
from langchain_openai import ChatOpenAI
from typing import Annotated,TypedDict
from langgraph.graph import StateGraph, END
#load env var
load_dotenv()

openai_key= os.getenv("OPENAI_API_KEY")
llm_name="gpt-5-nano"

client = OpenAI(api_key=openai_key)
model =ChatOpenAI(api_key=openai_key,model=llm_name)
#step 1: Build a basic chatbot
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages:Annotated[list,add_messages]

def bot(state:State):
    print(state["messages"])
    return {"messages":[model.invoke(state["messages"])]}

graph_builder = StateGraph(State)

graph_builder.add_node("bot",bot)

graph_builder.set_entry_point("bot")

graph_builder.set_finish_point("bot")

graph = graph_builder.compile()

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


