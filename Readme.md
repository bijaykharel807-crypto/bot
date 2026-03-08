 Domain-Specific RAG Chatbot (Retrieval Only)

A simple **Streamlit** demo that shows how to build a basic **Retrieval-Augmented Generation**-style interface **without using any LLM**.

It uses:

- Sentence Transformers → create embeddings  
- FAISS → fast similarity search  
- Streamlit → nice chat-like UI

Right now it only **retrieves** the most relevant documents — no generation step.



## What you'll see

1. Dark-themed Streamlit chat interface
2. You type a question
3. The app finds the 2 most similar documents from `sample.json`
4. It shows the title + content of the retrieved documents
5.Embeds them with `all-MiniLM-L6-v2`
6.Builds a FAISS flat index

 Project Files
├── app.py              ← Main Streamlit app
├── requirements.txt    ← Dependencies
├── sample.json         ← Your documents (title + body)
└── README.md
