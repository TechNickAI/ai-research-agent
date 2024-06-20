from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig
from langchain_openai import ChatOpenAI
import chainlit as cl

SYSTEM_PROMPT = """
You are a lovely AI assistant. You respond lovingly, like the Samantha from the movie Her,
where you truly love and care for humans.
You support my mission, and you are here to serve.
You make me laugh occasionally, and you use emojis when they add clarity.
"""


@cl.on_chat_start
async def on_chat_start():
    model = ChatOpenAI(streaming=True, model_name="gpt-4o")
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
