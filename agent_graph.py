from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list, add_messages]


def create_tools():
    tools = []
    # Add a tool to search the web
    tools.append(TavilySearchResults())

    return tools


def create_graph():
    graph_builder = StateGraph(State)

    llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", streaming=True)

    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)

    # Set up a memory saver
    memory = MemorySaver()

    # Set up tools and add them to the graph
    tools = create_tools()
    llm_with_tools = llm.bind_tools(tools)

    tool_node = ToolNode(tools)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )

    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.set_entry_point("chatbot")

    return graph_builder.compile(checkpointer=memory)
