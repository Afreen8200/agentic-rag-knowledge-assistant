
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Step 1: Read our document
file_path = Path("data/raw/gitlab_communication.txt")
text = file_path.read_text(encoding="utf-8")

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

print("Total chunks:", len(chunks))

# Step 3: Load the embedding model
model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# Step 4: Convert first 3 chunks into embeddings
embeddings = model.embed_documents(chunks[:3])

# Step 5: Display the results
print("Embeddings created:", len(embeddings))
print("Vector dimensions:", len(embeddings[0]))
print("First 5 values:", embeddings[0][:5])
