import os
from uuid import uuid4

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from pinecone import Pinecone

load_dotenv()


def ingest_documents():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("INDEX_NAME"))
    embedding = PineconeEmbeddings(model="llama-text-embed-v2", dimension=2048)
    vector_store = PineconeVectorStore(index=index, embedding=embedding)

    profile_loader = PyPDFLoader("context/Profile.pdf")
    profile_documents = profile_loader.load()

    life_loader = PyPDFLoader("context/Life.pdf")
    life_documents = life_loader.load()

    side_projects_loader = PyPDFLoader("context/Side projects.pdf")
    side_projects_documents = side_projects_loader.load()

    bike_and_run_loader = PyPDFLoader("context/Bike and run stats.pdf")
    bike_and_run_documents = bike_and_run_loader.load()

    documents = [
        *profile_documents,
        *life_documents,
        *side_projects_documents,
        *bike_and_run_documents,
    ]
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)

    return documents


if __name__ == "__main__":
    ingest_documents()
