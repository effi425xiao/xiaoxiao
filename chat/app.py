from operator import itemgetter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import (
    RunnableLambda,
    RunnablePassthrough,
)
from langchain.memory import ConversationBufferMemory
from langchain import hub
import chainlit as cl
from vdb_connect import get_latest_notes, vdb
from vdb_connect import retriever, retriever_chats
from vdb_connect import get_todays_date, format_docs
import time

load_dotenv()

model = ChatOpenAI(streaming=True, model="gpt-4-turbo-preview")


def store_chat_history(history: list, n: int = 4, vdb=vdb, namespace="past_chats"):
    # Ensure history has at least n items
    if len(history) < n:
        return

    # Get the last n items from history
    last_n_items = history[-n:]

    # Prepare the texts and metadatas lists
    texts = []
    metadatas = []

    # Prepare the metadata for the item
    the_time = get_todays_date()
    epochs = int(time.time())
    metadata = {"datetime": the_time, "epochs": epochs}

    # Combine the last n items into a single string
    combined_text = " ".join(
        [
            f"{'xiaofanli' if item.type == 'ai' else 'xiao'}: {item.content}"
            for item in last_n_items
        ]
    )
    combined_text += " " + "Date of chat: " + the_time
    texts.append(combined_text)
    metadatas.append(metadata)

    # Add the texts and metadatas to the vector database
    try:
        vdb.add_texts(texts, metadatas=metadatas, namespace=namespace)
    except Exception as e:
        print(f"Failed to store history: {e}")


prompt = hub.pull("xfanli/note_reader:bd4742ab")
retriever_chain = (
    {
        "context": itemgetter("query") | retriever | format_docs,
        "relevant_chats": itemgetter("query") | retriever_chats | format_docs,
        "query": itemgetter("query") | RunnablePassthrough(),
        "date": RunnableLambda(get_todays_date),
        "latest_notes": RunnableLambda(get_latest_notes),
        "": itemgetter("history") | RunnableLambda(store_chat_history),
        "history": itemgetter("history") | RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)


from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    session_id = cl.user_session.get("id")
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def setup_runnable(session_id):
    runnable = RunnableWithMessageHistory(
        retriever_chain,
        get_session_history,
        history_messages_key="history",
        input_messages_key="query",
    )
    cl.user_session.set("runnable", runnable)


avatar_url = "https://i.ibb.co/2j76LDW/logo-dark-copy.png"
avatar_url = "https://storage.googleapis.com/xfanli/xiaofanli2_small.png"

drop_image = "https://i.ibb.co/2j76LDW/logo-dark-copy.png"


@cl.on_chat_start
async def on_chat_start():
    # Set AI Avatar Image
    img_url = avatar_url
    await cl.Avatar(
        name="xiao fanli",
        url=img_url,
    ).send()

    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    setup_runnable(cl.user_session.get("id"))

    first_message = (
        "Hello! I'm xiao fanli, nice to meet you. \n\n"
        "![Image](https://storage.googleapis.com/xfanli/xiaofanli2_small.png)"
        "\n\n I might know you and have some notes for you. Place your chinese name here."
    )
    msg = cl.Message(content=first_message)

    await msg.send()


@cl.on_message
async def on_message(message: cl.Message):
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    runnable = cl.user_session.get("runnable")  # type: Runnable
    res = cl.Message(content="")

    session_id = cl.user_session.get("id")
    question_str = message.content
    async for chunk in runnable.astream(
        {"query": question_str},
        config={"configurable": {"session_id": session_id}},
    ):
        await res.stream_token(chunk)
    await res.send()


if __name__ == "__main__":
    cl.run("app.py")