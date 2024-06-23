from agent_graph import create_graph
from chainlit.input_widget import Select, Slider
from langchain.schema.runnable.config import RunnableConfig
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
    settings = await cl.ChatSettings(
        [
            Select(
                id="llm_provider",
                label="Language Model:",
                items={"OpenAI (GPT4o)": "openai", "Anthropic (Claude 3.5)": "anthropic"},
                initial_value="openai",
            ),
            Slider(
                id="temperature",
                label="Temperature:",
                initial=0.7,
                max=2,
                step=0.1,
                description="Higher setting will generate more random responses",
            ),
        ]
    ).send()
    cl.user_session.set("settings", settings)

    graph = create_graph(settings)
    cl.user_session.set("graph", graph)


@cl.on_settings_update
def on_settings_update(settings):
    cl.user_session.set("settings", settings)
    logger.info(f"Settings updated: {settings}")

    # Rebuild the graph
    graph = create_graph(settings)
    cl.user_session.set("graph", graph)


# Handle messages
@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("graph")  # type: Runnable
    settings = cl.user_session.get("settings")

    if settings["llm_provider"] == "anthropic":
        # Anthropic returns results that break the Chainlit callback handler
        callbacks = []
    else:
        callbacks = [cl.LangchainCallbackHandler(stream_final_answer=True)]

    msg = cl.Message(content="")

    config = RunnableConfig(callbacks=callbacks, configurable={"thread_id": message.thread_id})

    async for event in runnable.astream_events({"messages": message.content}, config=config, version="v1"):
        if event["event"] == "on_chat_model_stream":
            # Depending on the API (anthropic vs openai), the chunk is either a string or a list [with tool calls])
            if isinstance(event["data"]["chunk"].content, str):
                await msg.stream_token(event["data"]["chunk"].content)
            elif isinstance(event["data"]["chunk"].content, list):
                for chunk in event["data"]["chunk"].content:
                    if chunk["type"] == "text":
                        await msg.stream_token(chunk["text"])
                    elif chunk["type"] == "tool_use":
                        # It's a tool call
                        logger.info(f"Tool call: {chunk}")

    await msg.send()
