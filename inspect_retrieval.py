
# Import required libraries
import re
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# STEP 1: Load our embedding model
# MiniLM converts text into 384-dimensional vectors
model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)


# STEP 2: Connect to our existing ChromaDB
# We are using the database we created earlier
db = Chroma(
    collection_name="gitlab_handbook",
    embedding_function=model,
    persist_directory="data/chroma_db"
)


# STEP 3: Retrieve all stored document chunks
# These are the 370 chunks stored in ChromaDB
data = db.get(include=["documents"])

chunks = data["documents"]

if not chunks:
    raise ValueError("No documents found in ChromaDB")

print("Total stored chunks:", len(chunks))


# STEP 4: Create a tokenization function
# Convert text to lowercase and split it into words
# Example: "Slack is Down!" -> ["slack", "is", "down"]
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


# STEP 5: Initialize the BM25 keyword search model
# BM25 will rank chunks according to keyword relevance
tokenized_chunks = [
    tokenize(chunk) for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)


# STEP 6: Define questions for testing
# Each question has a target phrase we expect to find
questions = [
    (
        "What should employees do if Slack is down?",
        "slack is down"
    ),
    (
        "What should employees do if Zoom is down?",
        "zoom is down"
    )
]


# STEP 7: Test each question
for question, target in questions:

    print("\n" + "=" * 60)
    print("QUESTION:", question)
    print("=" * 60)


    # STEP 8: Perform semantic search using ChromaDB
    # Retrieve the top 10 semantically similar chunks
    semantic_results = db.similarity_search(
        question,
        k=10
    )

    print("\nSEMANTIC SEARCH - TOP 10")

    for rank, doc in enumerate(semantic_results, start=1):

        # Check whether the target phrase exists
        found = target in doc.page_content.lower()

        print(f"Rank {rank}: Target phrase found = {found}")

        # Display the matching document chunk
        if found:
            print("Matching chunk:")
            print(doc.page_content[:600])


    # STEP 9: Perform keyword search using BM25
    # Calculate keyword relevance scores
    query_tokens = tokenize(question)

    scores = bm25.get_scores(query_tokens)


    # STEP 10: Sort chunks by BM25 score
    # Higher scores represent stronger keyword matches
    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda i: scores[i],
        reverse=True
    )[:10]


    # STEP 11: Display the top 10 BM25 results
    print("\nBM25 KEYWORD SEARCH - TOP 10")

    for rank, index in enumerate(ranked_indices, start=1):

        # Get the matching chunk
        chunk = chunks[index]

        # Check for our target phrase
        found = target in chunk.lower()

        print(
            f"Rank {rank}: "
            f"Target phrase found = {found}, "
            f"BM25 Score = {scores[index]:.2f}"
        )

        # Display the chunk if the target is found
        if found:
            print("Matching chunk:")
            print(chunk[:600])


# STEP 12: Finish the evaluation
print("\nRetrieval comparison completed!")
