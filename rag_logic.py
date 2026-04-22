import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

class LocalRAG:
    def __init__(self):
        # Optimized for M1 Mac performance
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        self.llm = Ollama(model="llama3")
        self.vector_db_path = "vector_db"
        
        # Standard enterprise chunking strategy
        self.chunk_size = 1000
        self.chunk_overlap = 150

    def ingest_pdf(self, file_path):
        """Load, Split, and Vectorize documents into a persistent local store."""
        # 1. Document Ingestion
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # 2. Recursive Chunking (Preserves paragraph context)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_documents(docs)

        # 3. Vector Storage (Overwrites/Updates the existing DB)
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.vector_db_path
        )
        vector_db.persist()
        return len(chunks)

    def query(self, user_question):
        """Retrieve relevant context and generate an anchored response."""
        # Load the database from disk
        vector_db = Chroma(
            persist_directory=self.vector_db_path,
            embedding_function=self.embeddings
        )

        # THE MMR FIX: 
        # search_type="mmr" ensures the retriever looks for diverse snippets.
        # fetch_k=20 looks at more data, k=5 returns the most diverse 5 chunks.
        retriever = vector_db.as_retriever(
            search_type="mmr", 
            search_kwargs={'k': 5, 'fetch_k': 20}
        )

        # RetrievalQA 'stuff' chain anchors the LLM to your provided context
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        # Execute Query
        result = qa_chain.invoke({"query": user_question})
        
        # Return answer and the supporting source documents (for citations)
        return result["result"], result["source_documents"]

    def clear_db(self):
        """Utility to wipe the local vector storage for a clean slate."""
        import shutil
        if os.path.exists(self.vector_db_path):
            shutil.rmtree(self.vector_db_path)
        if os.path.exists("temp"):
            shutil.rmtree("temp")