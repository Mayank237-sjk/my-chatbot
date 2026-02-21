import os 
import streamlit as st
from dotenv import load_dotenv

#langchain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#step 1 : Page configuration
st.set_page_config(page_title="C++ RAG Chatbot",page_icon="☁️")
st.title("☁️ C++ RAG Chatbot ")
st.write("ask any related question related to c++")
 #step 2: LOad Environment variables
load_dotenv()

#step 3: Cache document loading
@st.cache_resource
def load_vector_store():
    #step A: load documents
    loader =TextLoader("C++_Introduction.txt",encoding='utf-8')
    documents=loader.load();
    #step B : split text
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=200,
    chunk_overlap=20)
    final_documents= text_splitter.split_documents(documents)
    embedding=HuggingFaceEmbeddings(model_name="all-miniLM-L6-v2")
    db=FAISS.from_documents(final_documents , embedding)
    return db    
    #step c: embeddings
    #step D: CReate FAISS Vector Store
db=load_vector_store()
user_input = st.text_input("Neter your question:")
if(user_input):
    document = db.similarity_search(user_input,k=3)
    st.subheader("<3 Retrieved context")
    for i,doc in enumerate(document):
        st.markdown(f"**Results {i+1}:**")
        st.write(doc.page_content)


