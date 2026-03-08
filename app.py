import streamlit as st
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    texts = []
    for item in data:
        texts.append('Title: ' + item['title'] + '\nContent: ' + item['body'])
    return texts

@st.cache_resource
def build_vector_store(texts):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return model, index

def retrieve_docs(query, embed_model, index, texts, k=2):
    query_vec = embed_model.encode([query])
    distances, indices = index.search(np.array(query_vec).astype('float32'), k)
    return [texts[i] for i in indices[0]]

def main():
    st.set_page_config(page_title='Domain RAG Chatbot', layout='wide')

    style = '<style>.stApp { background-color: #0e1117; color: white; }</style>'
    st.markdown(style, unsafe_allow_html=True)

    st.title('Domain-Specific RAG Chatbot')

    st.sidebar.title('Settings')
    st.sidebar.info('Retrieval only (no LLM)')

    try:
        texts = load_data('sample.json')
        embed_model, index = build_vector_store(texts)
    except FileNotFoundError:
        st.error('Error: sample.json not found in directory')
        return

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    if prompt := st.chat_input('Ask a question...'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.write(prompt)

        with st.chat_message('assistant'):
            with st.spinner('Searching...'):
                retrieved_context = retrieve_docs(prompt, embed_model, index, texts)
                context_str = '\n\n'.join(retrieved_context)

                # No LLM – just show the retrieved documents
                st.write("**Retrieved relevant documents:**")
                st.write(context_str)
                st.session_state.messages.append({'role': 'assistant', 'content': context_str})

if __name__ == '__main__':
    main()