#%%
import gradio as gr
import pdfplumber
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import ollama
import pytesseract
from PIL import Image

# Function to extract text with OCR as a fallback
def extract_text_with_ocr(page):
    text = page.extract_text()
    if not text:
        # Convert PDF page to an image
        image = page.to_image()
        # Perform OCR on the image
        text = pytesseract.image_to_string(image)
    return text

# Function to load, split, and retrieve documents from a PDF file
def load_and_retrieve_docs(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = extract_text_with_ocr(page)
                if page_text:
                    text += page_text
    except Exception as e:
        return f"Error reading PDF file: {e}"
    
    if not text:
        return "No text found in the PDF file."
    
    docs = [Document(page_content=text)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function that defines the RAG chain
def rag_chain(file, question):
    retriever = load_and_retrieve_docs(file)
    if isinstance(retriever, str):  # If an error message is returned
        return retriever
    
    retrieved_docs = retriever.get_relevant_documents(question)  # Use get_relevant_documents method
    formatted_context = format_docs(retrieved_docs)
    formatted_prompt = f"Question: {question}\n\nContext: {formatted_context}"
    response = ollama.chat(model='llama3', 
                           messages=[
                                {"role": "system",
                                 "content": "You are a helpful assistant. Check the pdf content and answer the question."
                                },
                                {"role": "user", "content": formatted_prompt}])
    return response['message']['content']

# Gradio interface
iface = gr.Interface(
    fn=rag_chain,
    inputs=["file", "text"],
    outputs="text",
    title="LLAMA 3: RAG Chain Question Answering",
    description="Upload a PDF and enter a query to get answers from the RAG chain."
)

# Launch the app
iface.launch()

# %%
