
import chromadb

# Step 1: Connect to our existing ChromaDB
client = chromadb.PersistentClient(
    path="data/chroma_db"
)

# Step 2: Open our GitLab collection
collection = client.get_collection(
    name="gitlab_handbook"
)

# Step 3: Get all stored document chunks
data = collection.get(include=["documents"])
documents = [doc for doc in data["documents"] if doc]

print("Total stored chunks:", len(documents))

# Step 4: Search for important information
search_terms = [
    "Slack is down",
    "Zoom is down",
    "Slack Down!",
    "Slack Huddle",
    "vacation",
    "paid time off"
]

# Step 5: Check each term
for term in search_terms:

    matches = [
        doc for doc in documents
        if term.lower() in doc.lower()
    ]

    print(f"\nSearching for: {term}")
    print("Matching chunks:", len(matches))

    if matches:
        example = matches[0]
        position = example.lower().find(term.lower())

        print("Example:")
        print(example[max(0, position-100):position+500])
