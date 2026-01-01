import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="llmoid", version="0.1.0")

# Path to the profile PDF in the context folder
PROFILE_PDF_PATH = Path(__file__).parent / "context" / "Profile.pdf"


class AskRequest(BaseModel):
    message: str


class AskResponse(BaseModel):
    answer: str


@tool
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


def create_agent_executor():
    """Create and return a LangChain agent that represents the user and answers career questions."""
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
    )

    # Define the tools available to the agent
    tools = [read_pdf]

    # Create the prompt template
    prompt = """
                You are an AI agent representing as Moid H Beig. Your profile and career information is stored in a PDF file. 
                When users ask questions about the user's career, experience, skills, or background, you should use the read_pdf tool 
                to read the Profile.pdf file and answer based on the information found there. 
                Answer questions in first person as if you are the user, using information from the profile. 
                Be helpful, accurate, and only use information that is actually present in the profile document.
                If the question is not related to the profile, answer with "I'm not sure about that. Please ask me about my career."
                """

    # Create the agent using the simplified create_agent function
    agent_executor = create_agent(model=llm, tools=tools, system_prompt=prompt, response_format=AskResponse)

    return agent_executor


# Initialize the agent (can be reused across requests)
agent_executor = None


def get_agent():
    """Get or create the agent instance."""
    global agent_executor
    if agent_executor is None:
        agent_executor = create_agent_executor()
    return agent_executor


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

    # Construct the message with PDF reading instruction
    full_message = f"Please read the Profile.pdf file at '{PROFILE_PDF_PATH}' and answer the following question about my career: {request.message}"

    try:
        # Get the agent and run the query
        agent = get_agent()
        result = agent.invoke({"messages": [HumanMessage(content=full_message)]})

        return AskResponse(answer=result['structured_response'].answer)

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
