
from pathlib import Path
import requests
from bs4 import BeautifulSoup

# Step 1: Website containing our document
url = "https://handbook.gitlab.com/handbook/communication/"

# Step 2: Download the webpage
response = requests.get(url, timeout=30)
response.raise_for_status()

# Step 3: Parse the HTML
soup = BeautifulSoup(response.text, "html.parser")

# Step 4: Find the main document
content = soup.find("article") or soup.find("main")

if content is None:
    raise ValueError("Document content not found")

# Step 5: Remove unnecessary elements
for element in content(["script", "style", "nav", "aside"]):
    element.decompose()

text = content.get_text(separator="\n", strip=True)

# Step 6: Create a folder to store documents
folder = Path("data/raw")
folder.mkdir(parents=True, exist_ok=True)

# Step 7: Save the document
file_path = folder / "gitlab_communication.txt"
file_path.write_text(
    f"Source: {url}\n\n{text}",
    encoding="utf-8"
)

print("Document downloaded successfully!")
print("Saved to:", file_path)
print("Characters:", len(text))
