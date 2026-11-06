
# Import required libraries
import requests
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# STEP 1: Download the GitLab handbook webpage
url = "https://handbook.gitlab.com/handbook/communication/"

response = requests.get(url, timeout=30)
response.raise_for_status()

# BeautifulSoup reads the HTML structure
soup = BeautifulSoup(response.text, "html.parser")

content = soup.find("article") or soup.find("main")

if content is None:
    raise ValueError("Could not find the handbook content")


# STEP 2: Remove unwanted HTML elements
for element in content(["script", "style", "nav", "aside"]):
    element.decompose()


# STEP 3: Group document text under its headings
sections = []

# Keep track of current headings (h1, h2, h3, h4...)
headings = {}

# Store the text belonging to the current section
current_lines = []


def save_section():
    # Save the section only if it contains text
    if not current_lines:
        return

    # Build a heading path, such as:
    # GitLab Communication > Emergency chat > Slack is down
    section_path = " > ".join(
        headings[level]
        for level in sorted(headings)
    )

    section_text = "\n".join(current_lines).strip()

    if section_text:
        sections.append((section_path, section_text))


# Read headings, paragraphs, lists, and code blocks in order
tags = [
    "h1", "h2", "h3", "h4", "h5", "h6",
    "p", "li", "pre"
]

for element in content.find_all(tags):

    # Check whether the element is a heading
    if element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:

        # Save the previous section
        save_section()

        current_lines = []

        # Identify heading level
        level = int(element.name[1])

        # Remove headings belonging to deeper levels
        headings = {
            key: value
            for key, value in headings.items()
            if key < level
        }

        # Store the new heading
        headings[level] = element.get_text(" ", strip=True)

        continue

    # Avoid duplicate paragraphs inside list items
    if element.name == "p" and element.find_parent("li"):
        continue

    # Avoid repeating nested list items
    if element.name == "li" and element.find_parent("li"):
        continue

    text = element.get_text(" ", strip=True)

    if not text:
        continue

    # Mark list instructions clearly
    if element.name == "li":
        text = "- " + text

    current_lines.append(text)


# Save the final section
save_section()

print("Sections extracted:", len(sections))


# STEP 4: Split only long sections
# Unlike our old approach, this does not mix headings
# from different sections.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

documents = []

for section_path, section_text in sections:

    # Divide long sections into smaller pieces
    pieces = splitter.split_text(section_text)

    for piece in pieces:

        # Keep the section heading in every chunk
        document = Document(
            page_content=(
                f"Section: {section_path}\n\n{piece}"
            ),
            metadata={
                "source": url,
                "section": section_path
            }
        )

        documents.append(document)

print("Section-aware chunks created:", len(documents))


# STEP 5: Verify our emergency sections
required_sections = [
    "Slack is down",
    "Zoom is down",
    "Slack and Zoom are down"
]

for heading in required_sections:

    matches = [
        doc for doc in documents
        if doc.metadata["section"].split(" > ")[-1].lower()
        == heading.lower()
    ]

    if not matches:
        raise ValueError(
            f"Required section missing: {heading}"
        )

    print(f"\n===== {heading} =====")

    for doc in matches:
        print(doc.page_content)


# STEP 6: Load our existing embedding model
# We are keeping MiniLM for a fair comparison
model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)


# STEP 7: Create a NEW ChromaDB knowledge base
# This location is separate from our original database
db = Chroma(
    collection_name="gitlab_handbook_sections",
    embedding_function=model,
    persist_directory="data/chroma_sections"
)


# STEP 8: Clear only this experiment's previous records
# Our original data/chroma_db remains untouched
old_ids = db.get()["ids"]

if old_ids:
    db.delete(ids=old_ids)


# STEP 9: Store section-aware chunks and embeddings
new_ids = [
    f"section-{i}"
    for i in range(len(documents))
]

db.add_documents(
    documents=documents,
    ids=new_ids
)


# STEP 10: Confirm storage
print("\nNEW KNOWLEDGE BASE CREATED!")
print("Collection:", "gitlab_handbook_sections")
print("Stored chunks:", db._collection.count())
print("Database folder:", "data/chroma_sections")
