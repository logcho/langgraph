from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]

llm = ChatOpenAI(model="gpt-4o")

def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state["messages"])
    state["messages"].append(response)
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

app = graph.compile()

if __name__ == "__main__":
    print("Chatbot started. Type 'exit' to quit.")
    state = {"messages": []}
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.strip().lower() == "exit":
                print("Goodbye!")
                break
            if not user_input.strip():
                continue
            
            state["messages"].append(HumanMessage(content=user_input))
            state = app.invoke(state)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

