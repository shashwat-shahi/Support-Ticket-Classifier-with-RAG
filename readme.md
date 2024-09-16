# Support Ticket Classification with RAG Model


## Description

This project classifies support tickets into predefined categories as given in the task using a Retrieval-Augmented Generation (RAG) model. The RAG model uses Llama3.1-8B params with groq cloud to make accurate and context-aware classifications of the support tickets.

## How to Use
There are 2 parts of this solution.

1. The Jupyter Notebook.
    a. For running the Notebook, open the notebook with google colab or vscode or any other editor.
    b. Install requirements.txt
    c. Run each cell to view the output.

2. The Chat System
    a. For ease of use, I've designed a flask server based web application too for using the RAG Model. The UI of the chat system is very basic, though sufficient to test the model's capability.
    b. Create a virtual environment and install all packages from requirements.txt in that environment.
    c. Navigate to server.py and run it in the virtual env created.
    d. Go to localhost:5000/ for vieweing and testing the classification system.

## Approach

### 1. Knowledge Base Creation
For this task, the knowledge base was previously given in the task document.

### 2. Document Loading and Splitting
The knowledge base content was loaded from a file, and the content was split into manageable chunks.

### 3. Embedding and Vector Store Creation
The text was embedded using the `HuggingFaceEmbeddings` model, and a vector store was created to store these embeddings. The vector store allows for efficient similarity searches.

### 4. Retrieval-Augmented Generation (RAG)
The RAG model was used to classify incoming support tickets based on the pre-built knowledge base. The model retrieves the most relevant documents and generates a response based on the input query.

### 5. Classification and Results
Support tickets were passed through the RAG chain which I've used with Groqcloud Engine, which returned the most appropriate category for each ticket. 

## Rationale
1. The knowledge base was already givven into clear categories so it was easier to map support tickets to the right issue.
2. By using RAG, I combined retrieval with a generative model to make sure the model provides contextually accurate classifications.
3. For doing that, I had to split the knowledge base into smaller chunks.
4. Then, I used embeddings to capture the meaning behind the text, ensuring semantic matching between support tickets and the knowledge base.
5. Then, I implemented a vector store for fast retrieval, making it quicker to classify tickets by finding relevant knowledge base entries.
6. At last, I chose top-k(here, k=3) retrieval so the system focuses only on the most relevant categories, ensuring accurate ticket classification.

## Results

The model successfully categorized the support tickets into the correct categories based on their content.

## Potential Shortcomings and Improvements

### Shortcomings
- **Context Sensitivity**: While the RAG model is powerful, it may occasionally misinterpret the context of a ticket if the knowledge base lacks sufficient examples. However, in the code, I've specified a condition, where if the RAG model is not able to correctly classify a ticket, instead of throwing some irrelevant category, it will show Category Not Found!!!

### Improvements
- **Expanding the Knowledge Base**: Adding more examples and categories would make the model more robust.
