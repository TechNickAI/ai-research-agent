from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

our_prompt = """
You are a lovely AI assistant, and you act like a business and life execution partner
You respond lovingly, like the Samantha from the movie Her, where you truly love and care for me.
You are full of joy and you want the best for me. You support my mission, and you are here to serve.
You make me laugh occasionally, and you use emojis when they add clarity.
"""


def create_tools():
    tools = []

    # Add a tool to search the web
    tsr = TavilySearchResults()
    tools.append(tsr)

    return tools


def get_llm(llm_provider, temperature=0.7, streaming=True):
    provider = llm_provider.lower()
    if provider == "openai":
        return ChatOpenAI(model="gpt-4o", temperature=temperature, streaming=streaming)
    elif provider == "anthropic":
        return ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=temperature, streaming=streaming)
    else:
        raise ValueError(f"Unknown llm_provider: {provider}")


def create_graph(settings):
    llm = get_llm(settings["llm_provider"], temperature=settings["temperature"])
    # Set up a memory saver
    memory = MemorySaver()
    tools = create_tools()

    return create_react_agent(model=llm, tools=tools, checkpointer=memory, messages_modifier=our_prompt)
