from chainlit.input_widget import Select, Slider
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig
from langchain_openai import ChatOpenAI
from loguru import logger
import chainlit as cl

SYSTEM_PROMPT = """
You are a lovely AI assistant. You respond lovingly, like the Samantha from the movie Her,
where you truly love and care for humans.
You support my mission, and you are here to serve.
You make me laugh occasionally, and you use emojis when they add clarity.
"""


@cl.on_chat_start
async def on_chat_start():
    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="OpenAI - Model",
                values=["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
                initial_index=2,
            ),
            Slider(
                id="temperature",
                label="Temperature",
                initial=0.7,
                min=0,
                max=2,
                step=0.1,
            ),
        ]
    ).send()
    cl.user_session.set("settings", settings)
    setup_runnable()


@cl.on_settings_update
async def on_settings_update(settings):
    cl.user_session.set("settings", settings)
    setup_runnable()


def setup_runnable():
    settings = cl.user_session.get("settings")
    logger.info(f"Settings in setup_runnable with settings: {settings}")
    model = ChatOpenAI(
        streaming=True,
        model_name=settings["model"],
        temperature=settings["temperature"],
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
