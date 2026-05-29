import os
from typing import TypedDict, List, Union
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv


load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    # print(response)
    print("AI: ", response.content)
    print("CURRENT STATE", state["messages"])
    return state
    
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
app = graph.compile()

conversation_history = []

user_input = input("Enter: ")

while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = app.invoke({"messages": conversation_history})
    # print(result["messages"])
    conversation_history = result["messages"]
    user_input = input("Enter: ")

with open ("loggin.txt", "w") as f:
    f.write("Your conversation history:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"User: {message.content}\n")
        elif isinstance(message, AIMessage):
            f.write(f"AI: {message.content}\n")
    f.write("\nEnd of conversation")

print("\nConversation history saved to loggin.txt")