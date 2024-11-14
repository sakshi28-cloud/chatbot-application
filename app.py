import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model Document Q&A")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
""")

# Function to handle the vector embeddings from the documents uploaded
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader(st.session_state.uploaded_folder)  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading

        if len(st.session_state.docs) < 4:
            st.error("Not enough documents loaded. Please ensure there are at least 4 documents.")
            return  # Exit the function if not enough documents

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:4])  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector OpenAI embeddings
        st.session_state.initialized = True

# Sidebar for PDF upload
st.sidebar.title("Upload PDF Documents")
uploaded_files = st.sidebar.file_uploader("Choose PDF files", accept_multiple_files=True, type=['pdf'])
if uploaded_files:
    upload_folder = './uploaded_docs'
    os.makedirs(upload_folder, exist_ok=True)
    for uploaded_file in uploaded_files:
        with open(os.path.join(upload_folder, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.session_state.uploaded_folder = upload_folder

prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Initialize Document Embeddings"):
    vector_embedding()
    if 'initialized' in st.session_state and st.session_state.initialized:
        st.success("Document embeddings have been initialized successfully.")
    else:
        st.error("Failed to initialize document embeddings. Please check the uploaded documents and try again.")

# Check if initialized before proceeding
if 'initialized' in st.session_state and st.session_state.initialized:
    if prompt1:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        print("Response time :", time.process_time() - start)
        st.write(response['answer'])

        # With a Streamlit expander
        with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
else:
    st.warning("Please initialize the document embeddings first by clicking 'Initialize Document Embeddings'.")




