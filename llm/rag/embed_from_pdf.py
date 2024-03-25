from langchain_community.document_loaders import PyPDFLoader
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


def get_path():
    parent_path = os.getcwd()
    docs_path = "/raw_data/pdf/"
    filename = "langchain_text1.txt"
    return parent_path + docs_path + filename


def data_loader():
    loader = PyPDFLoader("example_data/layout-parser-paper.pdf")
    pages = loader.load_and_split()


def main():
    pdf_file_path = get_path()
    data_loader()


if __name__ == "__main__":
    main()
