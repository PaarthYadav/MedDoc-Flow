import streamlit as st
from app.ui import pdf_uploader
from app.pdf_utils import extract_text_from_pdf, extract_text_from_txt
from app.vectorstore_utils import create_faiss_index, retrieve_relevant_docs
from app.chat_utils import get_chat_model, ask_chat_model
from app.config import EURI_API_KEY
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


st.set_page_config(
    page_title="MedDoc Flow - Medical Document Assistant",
    page_icon=":hospital:",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: lightblue;
        color: white;
    }
    .chat-message.assistant {
        background-color: #f0f2f6;
        color: black;
    }
    .chat-message .avatar {
        width: 2rem;
        height: 2rem;
        border-radius: 50%;
        margin-right: 0.5rem;
    }

    .chat-message .message {
        flex: 1;
    }

    .chat-message .timestamp {
        font-size: 0.8rem;
        opacity: 0.7;
        margin-top: 0.5rem;
    }
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #ff3333;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
            }
.status-success {
    background-color: #d4edda;
    color: #155724;
    padding: 0.5rem;
    border-radius: 0.25rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_model' not in st.session_state:
    st.session_state.chat_model = None
if 'doc_stats' not in st.session_state:
    st.session_state.doc_stats = None


st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="color: blue; font-size: 3rem; margin-bottom: 0.5rem;">🩺 MedDoc Flow</h1>
    <p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem;">Your Intelligent Medical Document Assistant</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for document upload
with st.sidebar:
    st.markdown("### 📄 Document Upload")
    st.markdown("Upload your medical documents to start chatting!")

    uploaded_files = pdf_uploader()

    if uploaded_files:
        st.success(f"📄 {len(uploaded_files)} document(s) uploaded")

        # Process documents
        if st.button("⚙️ Process Documents", type="primary"):
            with st.spinner("Processing your medical documents..."):
                all_texts = []
                errors = []

                for file in uploaded_files:
                    try:
                        if file.name.lower().endswith(".txt"):
                            text = extract_text_from_txt(file)
                        else:
                            text = extract_text_from_pdf(file)
                        all_texts.append(text)
                    except ValueError as exc:
                        errors.append(f"⚠️ {file.name}: {exc}")

                if errors:
                    for err in errors:
                        st.warning(err)

                if not all_texts:
                    st.error("❌ No text could be extracted from the uploaded files.")
                else:
                    # Split texts into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP,
                        length_function=len
                    )

                    chunks = []
                    for text in all_texts:
                        chunks.extend(text_splitter.split_text(text))

                    # Create FAISS index
                    vectorstore = create_faiss_index(chunks)
                    st.session_state.vectorstore = vectorstore

                    # Initialize chat model
                    try:
                        chat_model = get_chat_model(EURI_API_KEY)
                        st.session_state.chat_model = chat_model
                    except ValueError as exc:
                        st.error(f"❌ API key error: {exc}")
                        st.stop()

                    # Store document statistics
                    st.session_state.doc_stats = {
                        "files": len(uploaded_files) - len(errors),
                        "chunks": len(chunks),
                    }

                    st.success("✅ Documents processed successfully!")
                    st.balloons()

    # Document statistics
    if st.session_state.doc_stats:
        st.markdown("---")
        st.markdown("### 📊 Document Statistics")
        stats = st.session_state.doc_stats
        col1, col2 = st.columns(2)
        col1.metric("Files", stats["files"])
        col2.metric("Chunks", stats["chunks"])

    # Chat management
    st.markdown("---")
    st.markdown("### 🛠️ Chat Tools")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.messages:
        chat_export = "\n\n".join(
            f"[{msg['timestamp']}] {msg['role'].upper()}: {msg['content']}"
            for msg in st.session_state.messages
        )
        st.download_button(
            label="💾 Export Chat",
            data=chat_export,
            file_name="meddoc_chat_history.txt",
            mime="text/plain",
        )

# Main chat interface
st.markdown("### 💬 Chat with Your Medical Documents")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        st.caption(message["timestamp"])

# Chat input
if prompt := st.chat_input("Ask about your medical documents..."):
    # Add user message to chat history
    timestamp = time.strftime("%H:%M")
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": timestamp
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(timestamp)

    # Generate response
    if st.session_state.vectorstore and st.session_state.chat_model:
        with st.chat_message("assistant"):
            with st.spinner("🔎 Searching documents..."):
                # Retrieve relevant documents
                relevant_docs = retrieve_relevant_docs(st.session_state.vectorstore, prompt)

                # Create context from relevant documents
                context = "\n\n".join([doc.page_content for doc in relevant_docs])

                # Create prompt with context
                system_prompt = f"""You are MediChat Pro, an intelligent medical document assistant.
                Based on the following medical documents, provide accurate and helpful answers.
                If the information is not in the documents, clearly state that.

                Medical Documents:
                {context}

                User Question: {prompt}

                Answer:"""

                try:
                    response = ask_chat_model(st.session_state.chat_model, system_prompt)
                except RuntimeError as exc:
                    response = f"⚠️ Could not generate a response: {exc}"

            st.markdown(response)
            st.caption(timestamp)

            # Add assistant message to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": timestamp
            })
    else:
        with st.chat_message("assistant"):
            st.error("⚠️ Please upload and process documents first!")
            st.caption(timestamp)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
  <p>⚕️ Medical Document Intelligence</p>
</div>
""", unsafe_allow_html=True)