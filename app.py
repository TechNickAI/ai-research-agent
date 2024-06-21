from agent_graph import create_graph
import chainlit as cl


@cl.on_chat_start
async def on_chat_start():
    graph = create_graph()
    cl.user_session.set("graph", graph)


@cl.on_message
async def on_message(message: cl.Message):
    graph = cl.user_session.get("graph")

    msg = cl.Message(content="")

    # Call the graph with the message content and stream the response
    async for event in graph.astream_events({"messages": message.content}, version="v1"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            await msg.stream_token(chunk)

    await msg.send()
