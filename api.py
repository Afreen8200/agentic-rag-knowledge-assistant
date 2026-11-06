
# Import FastAPI and request validation
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Import LangGraph message types
from langchain_core.messages import HumanMessage, ToolMessage

# Import our existing compiled LangGraph agent
from langgraph_section_agent import graph


# STEP 1: Create the FastAPI application
app = FastAPI(
    title="Agentic RAG Knowledge Assistant",
    description="GitLab handbook assistant using LangGraph and RAG",
    version="1.0.0"
)


# STEP 2: Define what the user sends to the API
class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


# STEP 3: Create a simple health-check endpoint
@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Agentic RAG API is ready"
    }


# STEP 4: Create an endpoint for questions
@app.post("/ask")
def ask_question(request: QuestionRequest):

    print("\nAPI received question:", request.question)

    # Send the question to our LangGraph agent
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=request.question)
            ]
        },
        config={"recursion_limit": 8}
    )

    # Extract the final answer
    answer = result["messages"][-1].content

    # Check how many search tools executed
    tool_messages = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]

    # Return the answer as JSON
    return {
        "question": request.question,
        "answer": answer,
        "tool_executions": len(tool_messages)
    }
