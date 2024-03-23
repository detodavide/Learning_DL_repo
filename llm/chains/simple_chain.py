import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv, find_dotenv
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.globals import get_verbose

get_verbose()

load_dotenv(find_dotenv())

# Prompt template where to build you actual template
template = """You are an expert on machine learning and deep learning

{human_input}"""

# Generate PromptTemplate object used in the llm chain
prompt = PromptTemplate(input_variables=["human_input"], template=template)

# Get the LLM, in this case we are using the OpenAI gpt-3.5
llm = ChatOpenAI(verbose=True, temperature=0)

# Build the chain using the llm and the prompt
llm_input_chain = LLMChain(llm=llm, prompt=prompt, output_key="info")

prompt_info = PromptTemplate.from_template(
    template="So with this info: {info}. Can you go deeper about it?"
)

llm_info_chain = LLMChain(llm=llm, prompt=prompt_info)

# Call the API with this sequence of chains
sequence_chain = SequentialChain(
    chains=[llm_input_chain, llm_info_chain],
    input_variables=["human_input"],
    output_variables=["info"],
)

res = sequence_chain.invoke({"human_input": "Tell me about transformers"})
for k, v in res.items():
    print(f"key: {k} \nvalue: {v}")
