from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate


load_dotenv(find_dotenv())

parent_path = os.getcwd()
docs_path = "/raw_data/"
filename = "langchain_text1.txt"
relative_path = parent_path + docs_path + filename

with open(relative_path, "r") as file:
    text = file.read()

loader = TextLoader(relative_path)
docs = loader.load()
# docs = Document(page_content=text, metadata={"important_info": "langchain docs"})

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
)
documents = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()

# vectorstore = FAISS.from_documents(documents, embeddings)

# vectorstore.save_local("index")
vectorstore = FAISS.load_local(
    "index", embeddings, allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever()

prompt_template = """You are a helpful assistant for Langchain documentation.

{context}

Question: {question}
Answer here:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

llm = ChatOpenAI()
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs=chain_type_kwargs,
)

result = qa.invoke(input="What is indexing in langchain??")
print(result)
