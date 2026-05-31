import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Sequence, Annotated
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0) # minimize hallucinations

# embedding model has to be compatible with llm
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

pdf_path = "Stock_Market_Performance_2024.pdf"


if not os.path.exists(pdf_path):
    raise FileNotFoundError("PDF file not found")

try:
    pdf_loader = PyPDFLoader(pdf_path)
    docs = pdf_loader.load()
    print(f"Loaded {len(docs)} pages from PDF")
except FileNotFoundError as e:
    print(e)
    exit()

# chunking process
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

# split into chunks
chunks = text_splitter.split_documents(docs)

pages_split = text_splitter.split_documents(docs)

persist_directory = "./.chroma_db"
collection_name = "stock_market"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory, 
        collection_name=collection_name
    )

    print("Created ChromaDB Vector Store!")
except Exception as e:
    print(f"Error creating vector store: {e}")
    exit()

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

@tool
def retrieve_tool(query: str) -> str:
    """This tool searches and returns the information from the Stock Market Performance 2024 Document"""
    
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found. in the Stock Market Performance 2024 Document"
    
    results = []

    for i, doc in enumerate(docs):
        results.append(f"Document {i+1} - {doc.page_content}")
    
    return "\n\n".join(results)

tools = [retrieve_tool]
tool_dict = {t.name: t for t in tools}

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    """Checks if last message contains tool call"""
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

# LLM Agent
def call_llm(state: AgentState):
    """Calls the LLM with the current state"""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return { "messages": [message]}

# Retriever Agent
def take_action(state: AgentState):
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool {t['name']} with arguments {t['args']}")

        if not t['name'] in tool_dict:
            print(f"No tool found with name {t['name']}")
            result = "Incorrect Tool Call. Please try again."
        else:
            result = tool_dict[t['name']].invoke(t['args'])
            print(f"Result length: {len(result)}")
        
        results.append(ToolMessage(content=str(result), tool_call_id=t['id']))
    print(f"Tool execution completed")
    return {"messages": results}

graph = StateGraph(AgentState)

graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.set_entry_point("llm")

# Conditional Edge
graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True: "retriever_agent",
        False: END
    }
)

graph.add_edge("retriever_agent", "llm")
app = graph.compile()


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


def run_rag_agent():
    print("\n ----- STARTING AGENT ----")
    print("Type 'exit' to quit.")
    
    messages = []
    
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.strip().lower() == "exit":
                print("Goodbye!")
                break
            if not user_input.strip():
                continue
            
            messages.append(HumanMessage(content=user_input))
            state = {"messages": messages}
            
            printed_idx = len(messages) - 1
            
            for step in app.stream(state, stream_mode="values"):
                if "messages" in step:
                    msgs = step["messages"]
                    while printed_idx < len(msgs):
                        msgs[printed_idx].pretty_print()
                        printed_idx += 1
            
            # Save the final state's messages back to messages
            messages = step["messages"]
            
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
            
    print("\n ----- AGENT FINISHED ----")

if __name__ == "__main__":
    run_rag_agent()
    