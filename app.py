import streamlit as st
import os
from rag_logic import LocalRAG

# 1. Page Configuration & Stealth UI Styling
st.set_page_config(page_title="IntelRAG Service", layout="wide")

# Custom CSS for the Monochrome / Dark Mode aesthetic
st.markdown("""
    <style>
    .main { background-color: #000000; color: #ffffff; }
    .stButton>button { 
        background-color: #ffffff; color: #000000; 
        border-radius: 0px; font-weight: bold; width: 100%; 
    }
    .stTextInput>div>div>input { background-color: #111111; color: #ffffff; border: 1px solid #333; }
    .stChatMessage { background-color: #111111; border-radius: 0px; border-left: 3px solid #ffffff; }
    h1, h2, h3 { color: #ffffff; font-family: 'Courier New', Courier, monospace; }
    .stSidebar { background-color: #050505; border-right: 1px solid #222; }
    </style>
    """, unsafe_allow_html=True)

# 2. Initialize the Backend Engine
@st.cache_resource
def get_rag_engine():
    return LocalRAG()

rag = get_rag_engine()

# 3. Sidebar: Ingestion Management
with st.sidebar:
    st.title("📂 INTEL_CORE")
    st.write("---")
    uploaded_file = st.file_uploader("UPLOAD TECHNICAL DOCUMENT (PDF)", type="pdf")
    
    if uploaded_file:
        # Save file temporarily to disk for processing
        if not os.path.exists("temp"):
            os.makedirs("temp")
        
        file_path = os.path.join("temp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("VECTORIZING KNOWLEDGE..."):
            rag.ingest_pdf(file_path)
            st.success("INGESTION COMPLETE")
    
    st.write("---")
    if st.button("CLEAR VECTOR STORAGE"):
        # Logic to clear ChromaDB if needed
        st.warning("Database reset requested.")

# 4. Main Chat Interface
st.title("💬 LOCAL INTEL RAG")
st.subheader("Autonomous Knowledge Retrieval Engine")
st.write("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Input query..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("QUERYING LOCAL INTELLIGENCE..."):
            # Call our RAG backend
            answer, sources = rag.query(prompt)
            
            # Display response
            st.markdown(answer)
            
            # Display Citations (The "Source" Documents)
            with st.expander("VIEW SOURCE CITATIONS"):
                for doc in sources:
                    st.write(f"- **Page {doc.metadata.get('page', 'N/A')}**: {doc.page_content[:200]}...")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})