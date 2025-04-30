import streamlit as st
from langchain_community.document_loaders import (
    TextLoader, PDFPlumberLoader, UnstructuredWordDocumentLoader, UnstructuredHTMLLoader
)
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import os
import tempfile

# === Streamlit UI ===
st.set_page_config(page_title="Document Chatbot", layout="wide")
st.title("📄💬 Ask Your Document")

# === Initialize session states ===
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === Upload File ===
uploaded_file = st.file_uploader("Upload a file (.txt, .pdf, .docx, .html)", type=["txt", "pdf", "docx", "html"])

if uploaded_file is not None and st.session_state.qa_chain is None:
    try:
        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name

        # === Load document based on file extension ===
        def load_document(file_path):
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".txt":
                return TextLoader(file_path).load()
            elif ext == ".pdf":
                return PDFPlumberLoader(file_path).load()
            elif ext == ".docx":
                return UnstructuredWordDocumentLoader(file_path).load()
            elif ext == ".html":
                return UnstructuredHTMLLoader(file_path).load()
            else:
                raise ValueError(f"Unsupported file type: {ext}")

        docs = load_document(file_path)

        # === Split into chunks ===
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)

        # === Embeddings & Vector DB ===
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        retriever = vectorstore.as_retriever()

        # === LLM using OpenRouter ===
        llm = ChatOpenAI(
            model_name="mistralai/mistral-7b-instruct:free",
            temperature=0.7,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key="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"
        )
        st.session_state.qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        st.success("✅ Document processed! You can now chat below.")

    except Exception as e:
        st.error(f"❌ Error processing the file: {e}")

# === Chat Interface ===
if st.session_state.qa_chain is not None:
    st.subheader("Chat with your document")

    user_input = st.chat_input("Ask a question about the document")
    if user_input:
        response = st.session_state.qa_chain.invoke({"query": user_input})
        st.session_state.chat_history.append(("🧑", user_input))
        st.session_state.chat_history.append(("🤖", response["result"]))

    # Display chat history
    for sender, message in st.session_state.chat_history:
        with st.chat_message(sender):
            st.markdown(message)