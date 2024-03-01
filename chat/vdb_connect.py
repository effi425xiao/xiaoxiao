import os
from datetime import datetime
import pytz
import time
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone as pclient

load_dotenv()


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

if os.environ.get("PINECONE_API_KEY", None) is None:
    raise Exception("Missing `PINECONE_API_KEY` environment variable.")

pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PC_INDEX_NAME")

# Connect to the index
# Initialize Pinecone
pcd = pclient()
index = pcd.Index(os.getenv("PINECONE_INDEX_NAME"))
print(index.describe_index_stats())
print("*" * 100)
# Constants
HOURS_AGO = 12
TOP_K = 30

# Connect to the index

from langchain_pinecone import Pinecone

vdb = Pinecone(
    index=index_name,
    embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
    text_key="full_message",
    namespace="notes",
)

vdb_chat = Pinecone(
    index=index_name,
    embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
    text_key="full_message",
    namespace="past_chats",
)

# Get k=10 docs
retriever = vdb.as_retriever(search_kwargs={"k": 9})
retriever_chats = vdb_chat.as_retriever(search_kwargs={"k": 14})


class EmbeddingException(Exception):
    pass


def get_embedding(text: str) -> list:
    try:
        embd = embeddings.embed_query(text)
        return embd
    except Exception as e:
        print(f"Failed to get embedding: {e}")
        return None


def fetch_vectors(ids, index):
    return index._vector_api.fetch(ids)


def extract_and_sort_metadata(vectors):
    metadata_list = [vector["metadata"] for vector in vectors.values()]
    return sorted(metadata_list, key=lambda x: x["epochs"], reverse=True)


def format_notes(metadata_list):
    note_strings = [
        f"Date: {note['date']}\nFeelings: {note['feelings']}\nMessage: {note['full_message']}\n\n"
        for note in metadata_list
    ]
    return "".join(note_strings)


def query_index(emb, hours_ago, index):
    try:
        return index.query(
            vector=emb,
            top_k=TOP_K,
            namespace="notes",
            include_metadata=True,
            filter={"epochs": {"$gte": float(hours_ago)}},
        )
    except Exception as e:
        print(f"Failed to query index: {e}")
        return None


def calculate_hours_ago(hours_ago):
    return int(time.time()) - (hours_ago * 60 * 60)


def get_todays_date(input=""):
    # Get the current datetime with timezone
    now = datetime.now(pytz.timezone("America/Los_Angeles"))

    # Format the datetime as a string
    date_string = now.strftime("%Y-%m-%d %H:%M:%S %Z%z")

    return date_string


def get_latest_notes(input):
    hours_ago = calculate_hours_ago(HOURS_AGO)
    emb = get_embedding(get_todays_date())

    if emb is None:
        return "Couldn't reach the server to get the notes. Please try again later."

    query_result = query_index(emb, hours_ago, index)
    if query_result is None or query_result == []:
        return "Couldn't reach the server to get the notes. Please try again later."

    # print(f"Query result: {query_result}")

    # Extract metadata directly from query result
    metadata_list = [match["metadata"] for match in query_result["matches"]]
    metadata_list = sorted(metadata_list, key=lambda x: x["epochs"], reverse=True)

    for note in metadata_list:
        print(
            f"Note from {note['date']} ({note['epochs']}): {'within expected time range' if note['epochs'] >= hours_ago else 'outside expected time range'}"
        )

    print("\n" + "=" * 50)
    print(metadata_list)
    print("=" * 50 + "\n")

    return metadata_list


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


def main():
    print(get_latest_notes("chat from discord"))
    embed = get_embedding("just checking in")
    with open("txt.txt", "w") as file:
        file.write(str(embed))


if __name__ == "__main__":
    main()
