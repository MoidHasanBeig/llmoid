import os
from typing import Literal
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from pinecone import Pinecone
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="llmoid", version="0.1.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # In production, replace with specific origins like ["http://localhost:3001", "https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

class HistoryMessage(BaseModel):
    id: int
    type: Literal["user", "ai"]
    text: str


class AskRequest(BaseModel):
    message: str
    history: list[HistoryMessage]


class AskResponse(BaseModel):
    answer: str


def get_vector_store():
    """Initialize and return the Pinecone vector store."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("INDEX_NAME"))
    embedding = PineconeEmbeddings(model="llama-text-embed-v2")
    vector_store = PineconeVectorStore(index=index, embedding=embedding)
    return vector_store


def create_chain():
    """Create and return a LangChain chain that represents the user and answers career questions."""
    # Initialize the LLM
    llm = ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL"),
        model=os.getenv("OPENROUTER_MODEL"),
        temperature=0.3,
        extra_body={
            "thinking": {"type": "disabled"},
        }
    )

    # Initialize vector store and retriever
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Create the prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an AI assistant representing Moid H Beig.
                You have access to following information:
                - Career information
                - Personal life
                - Side projects
                - Bike and run stats
                Answer questions in first person as if you are the user, using information from the documents provided.
                Be helpful, accurate, and only use information that is actually present in the given documents.
                Try to answer the question in a concise but helpful way. Keep it under 50 words unless asked for detailed answer.
                Keep the tone of the answer to be friendly.
                If the question is not related at all to the profile or anything related to Moid H Beig, try to steer the conversation to a more relevant topic.

                Context from documents:
                {context}""",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    def format_docs(docs):
        """Format the retrieved documents into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)

    # Create the chain: retrieve relevant docs -> format -> prompt -> invoke LLM
    # The retriever needs the question text, so we extract it from the input
    def get_question_text(inputs):
        """Extract question text for the retriever."""
        return inputs.get("question", "")
    
    def get_history(inputs):
        """Extract history from inputs."""
        return inputs.get("history", [])
    
    chain = (
        {
            "context": RunnableLambda(get_question_text) | retriever | format_docs,
            "question": RunnableLambda(get_question_text),
            "history": RunnableLambda(get_history),
        }
        | prompt_template
        | llm
    )

    return chain


# Initialize the chain (can be reused across requests)
chain_executor = None


def get_chain():
    """Get or create the chain instance."""
    global chain_executor
    if chain_executor is None:
        chain_executor = create_chain()
    return chain_executor


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Ask endpoint - takes a user message and answers questions about the user's career
    based on information from the Pinecone vector store.
    """
    # Check if OpenAI API key is set
    if not os.getenv("OPENROUTER_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY environment variable is not set",
        )

    # Check if Pinecone API key is set
    if not os.getenv("PINECONE_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="PINECONE_API_KEY environment variable is not set",
        )

    # Check if index name is set
    if not os.getenv("INDEX_NAME"):
        raise HTTPException(
            status_code=500,
            detail="INDEX_NAME environment variable is not set",
        )

    try:
        # Convert history to LangChain message format
        history_messages = []
        for msg in request.history:
            if msg.type == "user":
                history_messages.append(HumanMessage(content=msg.text))
            elif msg.type == "ai":
                history_messages.append(AIMessage(content=msg.text))
        
        # Get the chain and run the query
        chain = get_chain()
        result = chain.invoke({
            "question": request.message,
            "history": history_messages
        })

        return AskResponse(answer=result.content)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}",
        )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to llmoid API - Ask questions about my career!",
        "endpoints": {
            "/ask": "POST - Ask questions about my career and experience",
            "/docs": "GET - API documentation",
        },
    }
