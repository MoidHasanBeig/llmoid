# llmoid

FastAPI application with a LangChain agent that represents you and answers questions about your career based on your profile.

## Setup

1. Install dependencies using `uv`:

```bash
uv sync
```

2. Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

## Running the Application

Start the FastAPI server:

```bash
uv run uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## Endpoints

### `POST /ask`

Ask questions about my career, experience, skills, or background. The agent will read the Profile.pdf from the context folder and answer based on that information.

**Request Body:**
```json
{
  "message": "What is your work experience?"
}
```

**Response:**
```json
{
  "answer": "I have worked as..."
}
```

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are your main skills?"
  }'
```

**Example using Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={
        "message": "Tell me about your career background"
    }
)
print(response.json()["answer"])
```

**Example Questions:**
- "What is your work experience?"
- "What skills do you have?"
- "Tell me about your education"
- "What projects have you worked on?"
- "What is your current role?"

### `GET /`

Root endpoint with API information.

## API Documentation

Once the server is running, you can access:
- Interactive API docs: `http://localhost:8000/docs`
- Alternative API docs: `http://localhost:8000/redoc`

## How It Works

1. The application uses LangChain with OpenAI to create an intelligent agent that represents you
2. Your profile information is stored in `context/Profile.pdf`
3. The agent has access to a `read_pdf` tool that can extract text from PDF files
4. When users ask questions about your career, the agent:
   - Reads the Profile.pdf from the context folder
   - Analyzes the content
   - Answers questions in first person based on the information in your profile

## Profile PDF

Make sure you have your `Profile.pdf` file in the `context/` folder. This file contains your career information, experience, skills, and background that the agent will use to answer questions.
