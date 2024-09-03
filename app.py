from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader 
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

api_key1 = os.getenv("GROK_API_KEY")
api_key2 = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="Learning Grok")
st.header("Learning Grok")

llm = ChatGroq(groq_api_key=api_key1, model="gemma-7b-it")

prompt = ChatPromptTemplate.from_template(
"""
You are an intelligent assistant with access to specific context information. Your task is to answer the following questions with accuracy and precision based on the provided context.

Context:
{context}

Questions:
{input}

Please ensure your responses are clear, concise, and directly relevant to the questions asked.
"""

)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./files")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

prompt1 = st.text_input("What do you want from the documents?")

if st.button("Creating Vector Store"):
    vector_embedding()
    st.write("Vector db has been created")

import time

if prompt1 and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retriever_chain.invoke({'input': prompt1})
    st.write(response['answer'])
