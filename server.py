import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from flask import Flask, request, jsonify, render_template

# Load environment variables
load_dotenv()



# Retrieve the API key from environment variables
groq_api_key = os.getenv('GROQ_API_KEY')


# Define classes to set up the knowledge base, model, vector store, and RAG chain
class KnowledgeBase:
    def __init__(self, content: List[str], file_path: str):
        self.content = content
        self.file_path = file_path

    def save_to_file(self):
        with open(self.file_path, "w") as file:
            for entry in self.content:
                file.write(entry + "\n")

# Define classes to set up the model and embeddings
class ModelSetup:
    def __init__(self, api_key: str, model_name: str):
        self.model = ChatGroq(api_key=api_key, model=model_name)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        

# Define classes to set up the vector store and RAG chain
class VectorStore:
    def __init__(self, file_path: str, embeddings):
        self.file_path = file_path
        self.embeddings = embeddings
        self.vectorstore = None

    def create_vectorstore(self):
        loader = TextLoader(self.file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
        splits = text_splitter.split_documents(docs)
        self.vectorstore = FAISS.from_documents(documents=splits, embedding=self.embeddings)

    def get_retriever(self, k: int = 3):
        return self.vectorstore.as_retriever(k=k)

    def similarity_search(self, query: str, k: int = 3):
        return self.vectorstore.similarity_search(query, k=k)
    

# Define classes to set up the RAG chain
class ChainSetup:
    def __init__(self, model, retriever):
        self.model = model
        self.retriever = retriever
        self.rag_chain = None

    def create_rag_chain(self):
        system_prompt = """
      # Guidelines

      ## Background Information:
      {context}

      ## Review the provided background information, which includes category labels and their descriptions. Then, follow the below steps:
      1. Analyze the given input text.
      2. Categorize the input into one of these categories:
        - Category 1 - Login Issues
        - Category 2 - App Functionality
        - Category 3 - Billing 
        - Category 4 - Account Management
        - Category 5 - Performance Issues

      3. If the input is unrelated to the background information or you're unsure of the classification, respond with 'Category Not Found!!!'.

      4. Provide only the category label as your response, without any additional text.
        """
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(self.model, qa_prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)

    def invoke_chain(self, input_text: str):
        return self.rag_chain.invoke({"input": input_text})

app = Flask(__name__)

# Global variables to store our setup
knowledge_base = None
model_setup = None
vector_store = None
chain_setup = None

def initialize_application():
    global knowledge_base, model_setup, vector_store, chain_setup

    # Define and save knowledge base
    knowledge_base_content = [
        "Category 1 - Login Issues - Login issues often occur due to incorrect passwords or account lockouts.",
        "Category 2 - App Functionality - App crashes can be caused by outdated software or device incompatibility.",
        "Category 3 - Billing - Billing discrepancies may result from processing errors or duplicate transactions.",
        "Category 4 - Account Management - Account management includes tasks such as changing profile information, linking social media accounts, and managing privacy settings.",
        "Category 5 - Performance Issues - Performance issues can be related to device specifications, network connectivity, or app optimization."
    ]
    file_path = "Data/knowledge_base.txt"
    knowledge_base = KnowledgeBase(knowledge_base_content, file_path)
    knowledge_base.save_to_file()

    # Set up model and embeddings
    # groq_api_key = groq_api_key
    model_setup = ModelSetup(groq_api_key, "llama-3.1-8b-instant")

    # Create vector store
    vector_store = VectorStore(file_path, model_setup.embeddings)
    vector_store.create_vectorstore()
    retriever = vector_store.get_retriever()

    # Set up and create RAG chain
    chain_setup = ChainSetup(model_setup.model, retriever)
    chain_setup.create_rag_chain()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify_ticket():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text in request'}), 400

    response = chain_setup.invoke_chain(data['text'])
    return jsonify({'category': response['answer']})

@app.route('/bulk_classify', methods=['POST'])
def bulk_classify_tickets():
    data = request.json
    if not data or 'tickets' not in data or not isinstance(data['tickets'], list):
        return jsonify({'error': 'Invalid request format'}), 400

    results = []
    for ticket in data['tickets']:
        if 'text' in ticket:
            response = chain_setup.invoke_chain(ticket['text'])
            results.append({
                'text': ticket['text'],
                'category': response['answer']
            })

    return jsonify({'results': results})


if __name__ == "__main__":
    initialize_application()
    app.run(debug=True)