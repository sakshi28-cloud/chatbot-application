import os
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
""")

# Setup document folder
uploaded_folder = './uploaded_docs'
if not os.path.exists(uploaded_folder):
    os.makedirs(uploaded_folder)

def vector_embedding():
    if "vectors" not in globals():
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        loader = PyPDFDirectoryLoader(uploaded_folder)
        docs = loader.load()

        if len(docs) < 4:
            raise ValueError("Not enough documents loaded. Please ensure there are at least 4 documents.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs[:4])
        vectors = FAISS.from_documents(final_documents, embeddings)
        globals()['vectors'] = vectors

# Main execution
def main():
    prompt1 = input("Enter Your Question From Documents: ")

    if input("Initialize Document Embeddings? (yes/no): ").lower() == 'yes':
        vector_embedding()
        if 'vectors' in globals():
            print("Document embeddings have been initialized successfully.")
        else:
            print("Failed to initialize document embeddings. Please check the uploaded documents and try again.")

    if 'vectors' in globals():
        if prompt1:
            document_chain = create_stuff_documents_chain(llm, prompt)
            global_vectors = globals().get('vectors')
            retriever = global_vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            print("Response time:", time.process_time() - start)
            print(response['answer'])

            # Display document similarity search results
            print("Document Similarity Search Results:")
            for i, doc in enumerate(response["context"]):
                print(doc.page_content)
                print("--------------------------------")
    else:
        print("Please initialize the document embeddings first by typing 'yes' when prompted.")

if __name__ == "__main__":
    main() 