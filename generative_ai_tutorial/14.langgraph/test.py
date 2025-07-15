from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from ddgs import DDGS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from langchain_google_genai import GoogleGenerativeAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
from pprint import pprint

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

os.environ["GOOGLE_API_KEY"] = "AIzaSyBcUsfH8V9z9ES0SVlYRAZAY_Lp2AdO800"

llm=GoogleGenerativeAI(
    model="gemini-2.5-flash", temperature=0.1
)

embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")

@tool
def addition(input: str) -> float:
    """Add two numbers. Input: 'a,b' (e.g., '4.0,5')"""
    a_str, b_str = input.split(",")
    return float(a_str.strip()) + float(b_str.strip())

@tool
def subtraction(input: str) -> float:
    """Subtract second number from first. Input: 'a,b'"""
    a_str, b_str = input.split(",")
    return float(a_str.strip()) - float(b_str.strip())

@tool
def multiplication(input: str) -> float:
    """Multiply two numbers. Input: 'a,b'"""
    a_str, b_str = input.split(",")
    return float(a_str.strip()) * float(b_str.strip())

@tool
def division(input: str) -> float:
    """Divide first number by second. Input: 'a,b'"""
    a_str, b_str = input.split(",")
    b = float(b_str.strip())
    return float(a_str.strip()) / b if b != 0 else float("inf")

@tool
def search_duckduckgo(query: str) -> str:
    """Search the web for current or general knowledge using DuckDuckGo."""
    with DDGS() as ddgs:
        results = ddgs.text(query)
        top_results = [r["body"] for r in results][:3]
        return "\n".join(top_results)


tools = [
    addition,
    subtraction,
    multiplication,
    division,
    search_duckduckgo,
]

from langchain.agents import initialize_agent, AgentType

llm_with_tools = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# def llm_call(state:AgentState) -> AgentState:
#     system_prompt = SystemMessage(content=
#         "You are an intelligent AI assistant."
#     )
#     response = llm_with_tools.invoke([system_prompt] + state["messages"])
#     return {"messages": [response]}

def llm_call(state: AgentState) -> AgentState:

    
    system_prompt = SystemMessage(content="You are an intelligent AI assistant.")
    response = llm_with_tools.invoke([system_prompt] + state["messages"])

    k={"messages": state["messages"] + [AIMessage(content=str(response))]}
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    pprint(k)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    return k

def decision(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState)

graph.add_node("agent", llm_call)
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    decision,
    {
        "continue": "tools",
        "end": END,
    },
)
graph.add_edge("tools", "agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        print("-------------------------------------------------------------------")
        print(s)
        print("-------------------------------------------------------------------")
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(message)
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        else:
            print("*****************************************************************")
            message.pretty_print()
            print("*****************************************************************")



# inputs = {"messages": [("user", "Add 40 and 12. Then multiply the result by 6.")]}

from langchain_core.messages import HumanMessage

inputs = {
    "messages": [HumanMessage(content="Find the result of adding 24 and 18, subtracting 10 from it, multiplying the result by 3, then dividing that by 2. Also, tell me what the capital of France is.")]
}

print_stream(app.stream(inputs, stream_mode="values"))