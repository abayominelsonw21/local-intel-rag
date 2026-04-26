import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import os

# Securely pull the API key from Streamlit's vault
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# --- Page Config ---
st.set_page_config(page_title="Enterprise RAG Engine", page_icon="🧠", layout="wide")

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

# Initialize chat history early so the UI loop can use it
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar: Document Upload ---
with st.sidebar:
    st.header("1. Secure Upload")
    pdf_docs = st.file_uploader("Upload PDF Documents", accept_multiple_files=True, type=['pdf'])
    
    if st.button("Process & Vectorize"):
        if not pdf_docs:
            st.error("Please upload a PDF document.")
        else:
            with st.spinner("Chunking text and generating vectors..."):
                # 1. Extract & Chunk Text
                raw_text = ""
                for pdf in pdf_docs:
                    pdf_reader = PdfReader(pdf)
                    for page in pdf_reader.pages:
                        raw_text += page.extract_text()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                text_chunks = text_splitter.split_text(raw_text)
                
                # 2. Create Vector Store & Retriever
                embeddings = OpenAIEmbeddings()
                vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
                retriever = vectorstore.as_retriever()

                llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

                # 3. Contextualize question (History Aware)
                contextualize_q_system_prompt = (
                    "Given a chat history and the latest user question "
                    "which might reference context in the chat history, "
                    "formulate a standalone question which can be understood "
                    "without the chat history. Do NOT answer the question, "
                    "just reformulate it if needed and otherwise return it as is."
                )
                contextualize_q_prompt = ChatPromptTemplate.from_messages([
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])
                history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

                # 4. Answer question chain
                qa_system_prompt = (
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer the question. "
                    "If you don't know the answer, just say that you don't know. "
                    "Use three sentences maximum and keep the answer concise.\n\n"
                    "{context}"
                )
                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", qa_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

                # 5. Combine into final LCEL RAG chain
                st.session_state.conversation = create_retrieval_chain(history_aware_retriever, question_answer_chain)
                st.session_state.chat_history = []  # Reset history on new upload
                st.success("Vectorization Complete. Engine Ready.")

# --- Main Area: Chat Interface ---

# Display previous messages
for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

if prompt := st.chat_input("Query the embedded document..."):
    if "conversation" in st.session_state:
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner("Retrieving context..."):
            # Invoke the LCEL chain
            response = st.session_state.conversation.invoke({
                "chat_history": st.session_state.chat_history,
                "input": prompt
            })
            
            answer = response["answer"]
            
            # Update history manually
            st.session_state.chat_history.append(HumanMessage(content=prompt))
            st.session_state.chat_history.append(AIMessage(content=answer))
            
            with st.chat_message("assistant"):
                st.markdown(answer)
    else:
        st.warning("Please upload a document and initialize the vectorizer first.")