from agent_graph import create_graph
from loguru import logger
import chainlit as cl


# Login with Google
@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: dict[str, str],
    default_user: cl.User,
) -> cl.User | None:
    return default_user


# Set up the chat
@cl.on_chat_start
async def on_chat_start():
    graph = create_graph()
    cl.user_session.set("graph", graph)


# Handle messages
@cl.on_message
async def on_message(message: cl.Message):
    graph = cl.user_session.get("graph")

    # Call the graph with the message content and stream the response
    async for event in graph.astream_events({"messages": message.content}, version="v1"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            msg = cl.Message(content="", author="OXAI 💙")

            chunk = event["data"]["chunk"]
            if isinstance(chunk.content, str):
                # Handle direct response
                await msg.stream_token(chunk.content)
            elif isinstance(chunk.content, list):
                # Handle tool call
                for item in chunk.content:
                    if item["type"] == "text":
                        await msg.stream_token(item["text"])
                    elif item["type"] == "tool_use":
                        # Process tool call (e.g., log it, call a function, etc.)
                        msg = cl.Message(content="", author=item["name"], type="tool")
                        tool_name = item["name"]
                        tool_args = item["input"]
                        # Example: log the tool call
                        logger.info(f"Tool call: {tool_name} with args {tool_args}")

    await msg.send()
