
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Read our downloaded document
file_path = Path("data/raw/gitlab_communication.txt")
text = file_path.read_text(encoding="utf-8")

# Step 2: Create a text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len
)

# Step 3: Split the text
chunks = splitter.split_text(text) 

# Step 4: Count the chunks
print("Total chunks:", len(chunks))

# Step 5: Display the first three chunks
for i, chunk in enumerate(chunks[:3], start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)
