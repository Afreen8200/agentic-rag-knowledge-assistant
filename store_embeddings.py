
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Step 1: Read document and source
raw_text = Path(
    "data/raw/gitlab_communication.txt"
).read_text(encoding="utf-8")

source_line, text = raw_text.split("\n\n", 1)
source_url = source_line.replace("Source: ", "", 1)

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

# Step 3: Attach source metadata to each chunk
documents = [
    Document(
        page_content=chunk,
        metadata={"source": source_url}
    )
    for chunk in chunks
]

# Step 4: Load embedding model
model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# Step 5: Create a persistent vector database
db = Chroma(
    collection_name="gitlab_handbook",
    embedding_function=model,
    persist_directory="data/chroma_db"
)

# Step 6: Store all chunks with stable IDs
ids = [f"communication-{i}" for i in range(len(documents))]
db.add_documents(documents=documents, ids=ids)

print("Chunks processed:", len(documents))
print("Embeddings stored in ChromaDB!")

# Step 7: Test semantic search
results = db.similarity_search(
    "How does GitLab communicate remotely?",
    k=2
)

for i, doc in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print(doc.page_content[:500])
    print("Source:", doc.metadata["source"])
