
# Import required libraries
import re
import chromadb
from rank_bm25 import BM25Okapi


# STEP 1: Connect to our existing vector database
client = chromadb.PersistentClient(
    path="data/chroma_db"
)

collection = client.get_collection(
    name="gitlab_handbook"
)


# STEP 2: Load the stored chunks and their IDs
data = collection.get(include=["documents"])

chunk_ids = data["ids"]
chunks = data["documents"]

# Create a dictionary for finding chunks by ID
chunk_lookup = dict(zip(chunk_ids, chunks))

print("Total chunks:", len(chunks))


# STEP 3: Create a simple tokenizer
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


# STEP 4: Initialize BM25 keyword search
tokenized_chunks = [
    tokenize(chunk) for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)


# STEP 5: Ask a question
question = input("\nAsk about GitLab: ")


# STEP 6: Calculate BM25 relevance scores
scores = bm25.get_scores(tokenize(question))


# STEP 7: Select the top 5 matching chunks
top_indices = sorted(
    range(len(chunks)),
    key=lambda i: scores[i],
    reverse=True
)[:5]


# STEP 8: Retrieve each match and its neighbors
# Our saved chunks have IDs like communication-0,
# communication-1, communication-2, etc.
seen = set()

for rank, index in enumerate(top_indices, start=1):

    chunk_id = chunk_ids[index]

    print(f"\n===== BM25 MATCH {rank} =====")
    print("Matched chunk ID:", chunk_id)
    print("BM25 score:", round(scores[index], 2))

    # Extract chunk number from its ID
    match = re.fullmatch(
        r"communication-(\d+)",
        chunk_id
    )

    if match is None:
        print("Unexpected chunk ID format")
        continue

    chunk_number = int(match.group(1))

    # Retrieve previous, current and next chunks
    for offset in [-1, 0, 1]:

        neighbor_number = chunk_number + offset

        if neighbor_number < 0:
            continue

        neighbor_id = f"communication-{neighbor_number}"

        # Avoid printing the same chunk repeatedly
        if neighbor_id in seen:
            continue

        text = chunk_lookup.get(neighbor_id)

        if text is not None:
            seen.add(neighbor_id)

            print(f"\n--- {neighbor_id} ---")
            print(text)


print("\nContext expansion completed!")
