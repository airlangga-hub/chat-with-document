import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables
load_dotenv()

def load_document(file_path: str):
    """Load and parse a PDF document."""
    loader = UnstructuredPDFLoader(file_path)
    return loader.load()

def init_vectorstore(documents):
    """Initialize the vector store with document chunks."""
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    return FAISS.from_documents(chunks, embeddings)

def create_chain(vectorstore):
    """Create a retrieval chain for question answering."""
    llm = ChatGroq(model="llama3-70b-8192", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the following question ONLY based on the provided context, DO NOT answer outside the provided context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)

# Streamlit UI setup
st.set_page_config("Chat with your document")
st.title("Chat with your document! 💬")

# Initialize session state with chat history in the correct format from the start
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Will store tuples of (role, content)

# File upload sidebar
with st.sidebar:
    st.subheader("Upload your PDF document")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Process uploaded file
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        file_path = tmp_file.name
    
    try:
        if "vectorstore" not in st.session_state:
            with st.spinner("Analyzing document content..."):
                documents = load_document(file_path)
                st.session_state.vectorstore = init_vectorstore(documents)

        if "conversation_chain" not in st.session_state:
            st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)
        
        st.success("Document processed successfully! You can now ask questions about it.")
    finally:
        os.unlink(file_path)

# Display chat history
for role, content in st.session_state.chat_history:
    with st.chat_message("user" if role == "human" else "assistant"):
        st.text(content)

# Handle user input
user_input = st.chat_input("Ask a question about your document")

if user_input:
    if "vectorstore" not in st.session_state:
        st.error("Please upload a PDF document first before asking questions.")
    else:
        # Display user message
        with st.chat_message("user"):
            st.text(user_input)

        # Generate and display AI response
        with st.chat_message("assistant"):
            response = st.session_state.conversation_chain.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history
            })
            
            st.text(response["answer"])
        
        # Append chat history
        st.session_state.chat_history.append(("human", user_input))
        st.session_state.chat_history.append(("assistant", response["answer"]))