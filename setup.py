import streamlit as st
import os
from pathlib import Path
from langchain_pinecone import PineconeVectorStore
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import io
import contextlib

# API keys setup
st.write("OPENAI_API_KEY:", st.secrets["OPENAI_API_KEY"])
st.write("PINECONE_API_KEY:", st.secrets["PINECONE_API_KEY"])
st.write(
    "Has environment variables been set:",
    os.environ["OPENAI_API_KEY"] == st.secrets["OPENAI_API_KEY"],
    os.environ["PINECONE_API_KEY"] == st.secrets["PINECONE_API_KEY"]
)

PINECONE_ENV = "us-east-1"
index_name = 'ecl-rag'
dim = 1024

# Pinecone initialization
pinecone = Pinecone(api_key=os.getenv('PINECONE_API_KEY'), environment=PINECONE_ENV)
index = pinecone.Index(index_name)

# OpenAI embeddings
embeddings_1024 = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)


# Function to create chunks from a document
def create_chunks(doc_to_chunk):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_documents(doc_to_chunk)


# Function to load a PDF and create text chunks
def load_pdf(path):
    loader = PyPDFLoader(path)
    return loader.load()


def load_chunk_file(path):
    doc = load_pdf(path)
    return create_chunks(doc)


# Streamlit UI components for document processing
st.title('Document Processing and RAG System')

uploaded_file = st.file_uploader("Choose a file", type=['pdf'])
if uploaded_file is not None:
    file_path = Path(uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("File uploaded successfully!")
    if st.button('Load and Push to Pinecone'):
        chunks = load_chunk_file(file_path)
        # st.write("Processed Chunks:")
        # for chunk in chunks:
        #     st.text(chunk)
        # # Load the document into our database
        docsearch = PineconeVectorStore.from_documents(chunks, embeddings_1024, index_name=index_name)
        st.success(f"N° Chunks: {len(chunks)} uploaded to Pinecone successfully!")

# Prompt template for the RAG system
prompt_template = """
Please respond to the question as long as it pertains to ECL, Roxie, or querying. If asked to write code please write code in ECL syntax. Use the context to assist in answering the question but don't only use the context.

Use the following structure for your response:
\n\n
  Context:\n {context}\n
  Question: \n{question}\n


  Answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


# Function to handle data querying
def data_querying(question):
    vstore = PineconeVectorStore.from_existing_index(index_name, embeddings_1024)
    model_name = "gpt-4"
    llm = ChatOpenAI(model_name=model_name, temperature=0.3)
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

    docs = vstore.similarity_search(question, 10)
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    response_text = response.get('output_text')

    return response_text


st.title("Test PGM RAG GPT-4")

# Initialize chat history if not already in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
user_question = st.chat_input("Enter your question:")
if user_question:
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_question)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_question})

    # Generate response using the data querying function
    response_text = data_querying(user_question)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response_text)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})


def run_user_code(code):
    # Create a string buffer to capture the outputs
    output_buffer = io.StringIO()

    # Define a whitelist of safe modules and their submodules
    safe_modules = {
        'pgmpy': 'pgmpy',
        'numpy': 'numpy',  # pgmpy depends on numpy
    }

    # Restricted global environment with selected safe built-ins and safe module imports
    safe_globals = {
        "__builtins__": {
            "print": print,
            "range": range,
            "int": int,
            "float": float,
            "__import__": __import__,  # Allow __import__ for module loading
        }
    }

    # Allow importing only the specified safe modules, including submodules
    def custom_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Split module name to check for submodules
        base_module = name.split('.')[0]
        if base_module in safe_modules:
            # Use the built-in __import__ to process the import correctly
            return __import__(name, globals, locals, fromlist, level)
        else:
            raise ImportError(f"Import of {name} is not allowed")

    safe_globals['__builtins__']['__import__'] = custom_import

    try:
        # Redirect stdout to the buffer
        with contextlib.redirect_stdout(output_buffer):
            exec(code, safe_globals)  # Execute in a controlled environment
    except Exception as e:
        return f"An error occurred: {str(e)}"

    # Get the content of the output buffer
    return output_buffer.getvalue()


user_code = st.sidebar.text_area("Enter your code to execute:")
if st.sidebar.button('Run Code'):
    result = run_user_code(user_code)
    st.text_area("Code Output:", result, height=300)
