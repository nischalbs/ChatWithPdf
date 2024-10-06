# Import required modules
from langchain import hub
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
#from langchain.llms import Ollama
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
#from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
#from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os
import time

FILEPATH    = "sample.pdf" 
LOCAL_MODEL = "llama2"
EMBEDDING   = "nomic-embed-text"

loader = PyPDFLoader(FILEPATH)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=100)
all_splits = text_splitter.split_documents(data)

persist_directory = 'data'

vectorstore = Chroma.from_documents(
    documents=all_splits, 
    embedding=OllamaEmbeddings(model=EMBEDDING),
    persist_directory=persist_directory
)

# vectorstore.persist()

llm = Ollama(base_url="http://localhost:11434",
                                  model=LOCAL_MODEL,
                                  verbose=True,
                                  callback_manager=CallbackManager(
                                      [StreamingStdOutCallbackHandler()])
                                  )

retriever = vectorstore.as_retriever()

template = """ You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.
    
    Context: {context}
    History: {history}

    User: {question}
    Chatbot:
    """
prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )

memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question"
    )

qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": prompt,
                "memory": memory,
            }
        )

#query = input("Enter a Query")
#query = "mention the Tech Stack used"
query = "what does agglomerative cluster do"
#query = "which country has highest mortality"
query += ". Only from this pdf. Keep it short"
response = qa_chain(query)
#qa_chain.invoke({"query": query})
