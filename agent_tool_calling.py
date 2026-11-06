
# Import required libraries
from langchain_core.tools import tool
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage
)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


# STEP 1: Load the existing embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)


# STEP 2: Connect to our existing ChromaDB
db = Chroma(
    collection_name="gitlab_handbook",
    embedding_function=embedding_model,
    persist_directory="data/chroma_db"
)


# STEP 3: Create our first tool
# @tool makes this Python function available to the LLM
@tool
def search_gitlab_handbook(query: str) -> str:
    """Search the GitLab Communication Handbook for relevant information."""

    print("\nTOOL EXECUTED: Searching ChromaDB...")

    # Retrieve three semantically similar chunks
    documents = db.similarity_search(query, k=3)

    if not documents:
        return "No relevant documents found."

    # Format retrieved chunks for the LLM
    results = []

    for i, doc in enumerate(documents, start=1):

        results.append(
            f"[Chunk {i}]\n"
            f"{doc.page_content}\n"
            f"Source: {doc.metadata.get('source', 'Unknown')}"
        )

    return "\n\n".join(results)


# STEP 4: Connect to our local Llama model
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0
)


# STEP 5: Tell Llama which tools are available
# Binding a tool does not execute it
llm_with_tools = llm.bind_tools([
    search_gitlab_handbook
])


# STEP 6: Ask the user a question
question = input("\nAsk about GitLab: ")


# STEP 7: Prepare instructions and user message
messages = [
    SystemMessage(
        content=(
            "You are a GitLab knowledge assistant. "
            "For questions about GitLab policies, "
            "request the search_gitlab_handbook tool "
            "before answering. "
            "Use only retrieved evidence to answer. "
            "If evidence is insufficient, say you don't know."
        )
    ),
    HumanMessage(content=question)
]


# STEP 8: Ask Llama whether it wants to use a tool
first_response = llm_with_tools.invoke(messages)

print("\nTOOL CALLS REQUESTED:")
print(first_response.tool_calls)


# STEP 9: Execute any tools requested by Llama
if not first_response.tool_calls:

    print("\nLlama did not request a tool.")
    print("No grounded answer generated in this test.")

else:

    # Add Llama's tool request to the conversation
    messages.append(first_response)

    for tool_call in first_response.tool_calls:

        # Only execute our approved search tool
        if tool_call["name"] != search_gitlab_handbook.name:
            raise ValueError("Unknown tool requested")

        # Run the actual Python search function
        tool_result = search_gitlab_handbook.invoke(
            tool_call["args"]
        )

        print("\nTOOL RESULT:")
        print(tool_result)

        # Return the tool result to Llama
        messages.append(
            ToolMessage(
                content=tool_result,
                tool_call_id=tool_call["id"]
            )
        )

    # STEP 10: Ask Llama to answer using the tool result
    final_response = llm.invoke(messages)

    print("\nFINAL AI ANSWER:")
    print(final_response.content)
