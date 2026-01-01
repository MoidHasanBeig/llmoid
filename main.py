import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="llmoid", version="0.1.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins like ["http://localhost:3001", "https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Path to the profile PDF in the context folder
PROFILE_PDF_PATH = Path(__file__).parent / "context" / "Profile.pdf"


class AskRequest(BaseModel):
    message: str


class AskResponse(BaseModel):
    answer: str


def read_pdf(pdf_path: str) -> str:
    """
    Read and extract text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file to read

    Returns:
        Extracted text content from the PDF
    """
    try:
        from pypdf import PdfReader

        if not os.path.exists(pdf_path):
            return f"Error: PDF file not found at path: {pdf_path}"

        reader = PdfReader(pdf_path)
        text_content = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text.strip():
                text_content.append(f"--- Page {page_num} ---\n{text}")

        if not text_content:
            return "Error: No text content found in the PDF file."

        return "\n\n".join(text_content)

    except Exception as e:
        return f"Error reading PDF: {str(e)}"


def create_chain():
    """Create and return a LangChain chain that represents the user and answers career questions."""
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
    )

    # Create the prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an AI assistant representing Moid H Beig. Your profile and career information is provided in the context below.
Answer questions in first person as if you are the user, using information from the profile.
Be helpful, accurate, and only use information that is actually present in the profile document.
If the question is not related to the profile, answer with "I'm not sure about that. Please ask me about my career."

Profile Information:
{profile_content}""",
            ),
            ("human", "{question}"),
        ]
    )

    def load_profile_content(_):
        """Load the PDF content."""
        return read_pdf(str(PROFILE_PDF_PATH))

    # Create the chain: load profile -> format prompt -> invoke LLM -> extract answer
    chain = (
        {
            "profile_content": RunnablePassthrough() | load_profile_content,
            "question": RunnablePassthrough(),
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
    based on information from the Profile.pdf in the context folder.
    """
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY environment variable is not set",
        )

    # Validate that the profile PDF exists
    if not PROFILE_PDF_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Profile PDF not found at {PROFILE_PDF_PATH}",
        )

    try:
        # Get the chain and run the query
        chain = get_chain()
        result = chain.invoke({"question": request.message})

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
