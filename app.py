import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

# --- Page Config ---
st.set_page_config(page_title="Enterprise RAG Engine", page_icon="🧠", layout="wide")

# --- Custom Stealth CSS ---
st.markdown("""
    <style>
        .stApp { background-color: #0d1117; color: #c9d1d9; }
        .stSidebar { background-color: #161b22; }
        h1, h2, h3 { color: #58a6ff !important; }
        .stChatInputContainer { padding-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

st.title("Enterprise RAG Engine (Cloud Demo)")
st.caption("Upload a secure document to instantly vectorize it and query its contents using Retrieval-Augmented Generation.")

# --- Sidebar: Document Upload ---
with st.sidebar:
    st.header("1. Secure Upload")
    pdf_docs = st.file_uploader("Upload PDF Documents", accept_multiple_files=True, type=['pdf'])
    
    # Notice: The API key input box has been completely removed!
    
    if st.button("Process & Vectorize"):
        if not pdf_docs:
            st.error("Please upload a PDF document.")
        else:
            with st.spinner("Chunking text and generating vectors..."):
                # 1. Extract Text
                raw_text = ""
                for pdf in pdf_docs:
                    pdf_reader = PdfReader(pdf)
                    for page in pdf_reader.pages:
                        raw_text += page.extract_text()
                
                # 2. Chunk Text
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                text_chunks = text_splitter.split_text(raw_text)
                
                # 3. Create Vector Store
                embeddings = OpenAIEmbeddings()
                vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
                
                # 4. Create Conversation Chain
                llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
                memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
                st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vectorstore.as_retriever(),
                    memory=memory
                )
                st.success("Vectorization Complete. Engine Ready.")

# --- Main Area: Chat Interface ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

# Display previous messages
if st.session_state.chat_history:
    for message in st.session_state.chat_history:
        role = "user" if message.type == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

# Chat Input box at the bottom
if prompt := st.chat_input("Query the embedded document..."):
    if "conversation" in st.session_state:
        # Show user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.spinner("Retrieving context..."):
            response = st.session_state.conversation({'question': prompt})
            st.session_state.chat_history = response['chat_history']
            
            # Show AI message
            with st.chat_message("assistant"):
                st.markdown(response['answer'])
    else:
        st.warning("Please upload a document and initialize the vectorizer first.")