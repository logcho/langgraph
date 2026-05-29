from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage # The foundational class for all messages
from langchain_core.messages import ToolMessage # Passes back the content of a tool back to llm
from langchain_core.messages import SystemMessage # Messsage for providing information to the llm
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Annotated - provides additional context without affecting the type itself
# email = Annotated[str, "The email address of the user"]
# print(email.__metadata__)

# Sequence - to automatically handle the state updates for sequences such as adding messages to chat history

# Reducer function
# Rule that controls how updates from nodes are combined with existing state
# Tells us how to merge new values into existing state

# Without a reducer, updates would overwrite the existing state with the new values

# Without a reducer:
# messages: ["hi"]
# messages: ["Nice to meet you"] state is overwritten

# With a reducer (add_messages): The messages are appended
# messages: ["hi"]
# messages: ["hi", "Nice to meet you"]

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int) -> int:
    """This is an additiona function that adds 2 numbers together"""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """This is a subtraction function that subtracts 2 numbers together"""
    return a - b

@tool
def divide(a: int, b: int) -> int:
    """This is a division function that divides 2 numbers together"""
    return a / b

@tool
def multiply(a: int, b: int) -> int:
    """This is a multiplication function that multiplies 2 numbers together"""
    return a * b

tools = [add, subtract, divide, multiply]

model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
        "You are my AI assistant, please answer my query to the best of your ability."
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tools_node = ToolNode(tools)
graph.add_node("tools", tools_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 40 + 12 then multiply the result by 6. Also tell me a joke please")]}
print_stream(app.stream(inputs, stream_mode="values"))