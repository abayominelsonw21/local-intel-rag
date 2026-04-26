# 💬 LOCAL INTEL RAG
**Privacy-First Autonomous Knowledge Retrieval Engine**

A production-ready RAG (Retrieval-Augmented Generation) pipeline optimized for M1 Mac Silicon. This system allows for 100% local, private document querying without external API dependencies.

### 🛠 Tech Stack
- **Orchestration:** LangChain
- **Vector Storage:** ChromaDB
- **Models:** Ollama (Llama 3 for Inference, mxbai-embed-large for Embeddings)
- **Interface:** Streamlit (Stealth Monochrome UI)

### 🚀 Key Engineering Features
- **MMR Retrieval:** Implements Maximal Marginal Relevance to ensure high-diversity context windows, preventing retrieval bias in multi-page documents.
- **Persistent Vector Store:** Local disk-based storage for stateful document memory.
- **Source Citations:** Full transparency with anchored citations to prevent AI hallucinations.
- **Hardware Optimized:** Leverages Apple's Unified Memory for low-latency local inference.

### 🧪 Tested Use Case
Successfully parsed complex 3-page professional resumes to retrieve specific performance metrics (e.g., 15% improvement in processing performance at Epsilon) with 100% accuracy and zero hallucination.
