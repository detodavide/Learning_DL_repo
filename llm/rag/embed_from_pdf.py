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
    filename = "mamba_linear-time-sequence.pdf"
    return parent_path + docs_path + filename


def main():
    pdf_file_path = get_path()
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
    )
    documents = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    # vectorstore = FAISS.from_documents(documents, embeddings)

    # vectorstore.save_local("mamba_index")

    vectorstore = FAISS.load_local(
        "mamba_index", embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever()

    prompt_template = """You are a helpful assistant for Mamba architecture paper documentation.

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

    result = qa.invoke(input="How mamba works?")
    print(result)


if __name__ == "__main__":
    main()
