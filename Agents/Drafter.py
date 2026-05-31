from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
@tool
def update(content: str) -> str:
    """Append the content to the global document string"""
    global document_content
    document_content = content
    return f"Document updated the current content is: {document_content}"

@tool
def save(filename: str) -> str:
    """Save the current document to a file
    Args:
        filename (str): The name of the file to save the document to
    
    """
    global document_content
    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"
    try:
        with open(filename, "w") as f:
            f.write(document_content)
        return f"Document saved as {filename}"
    except Exception as e:
        return f"Error saving document: {e}"
        
tools = [update, save]

model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content=f"""
        You are Drafter, a helpful writing assistant that drafts documents and saves the output to a file.
        Call the 'update' function to update the global document string with the current draft.
        Call the 'save' function to save the current document to a file.
        The current document is: {document_content}
        """
    )
    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like to create"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("User: ")
        user_message = HumanMessage(content=user_input)
    
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    response = model.invoke(all_messages)
    
    print(f"AI: {response.content}\n")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"USING TOOLS: {[tc[ 'name'] for tc in response.tool_calls]}\n")

    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state: AgentState):
    """Determines whether the conversation should continue or end"""
    messages = state["messages"]

    if not messages:
        return "continue"

    for message in reversed(messages):
        if isinstance(message, ToolMessage) and "saved" in message.content.lower() and "document" in message.content.lower():
            return "end"
        
    return "continue"
        
def print_messages(messages):
    """Function made to print messages more readably"""
    if not messages:
        return

    tool_messages = []
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            tool_messages.append(message)
        else:
            break
    for message in reversed(tool_messages):
        print(f"TOOL RESULT: {message.content}")

graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")
graph.add_conditional_edges(
    "tools", 
    should_continue,
    {
        "continue": "agent",
        "end": END
    }
)

app = graph.compile()

def run_document_agent():
    print("\n ----- STARTING AGENT ----")
    state = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if("messages" in step):
            print_messages(step["messages"])
    print("\n ----- AGENT FINISHED ----")

if __name__ == "__main__":
    run_document_agent()
    
