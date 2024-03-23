from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
import os

load_dotenv(find_dotenv())

parent_path = os.getcwd()
docs_path = "/docs/"
filename = "langchain_text1.txt"
relative_path = parent_path + docs_path + filename

with open(relative_path, "r") as file:
    text = file.read()

# loader = TextLoader(relative_path)
# docs = loader.load()
example_doc = Document(page_content=text, metadata={"important_info": "langchain docs"})

print(example_doc)
