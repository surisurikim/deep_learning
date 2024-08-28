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

# PDF page에서 텍스트 추출하는 함수 작성
def extract_text_with_ocr(page):
    text = page.extract_text()
    if not text: # 만약 추출할 텍스트가 없다면
        # PDF page를 이미지로 변환
        image = page.to_image()
        # 이미지에서 OCR 재실행하여 텍스트 추출
        text = pytesseract.image_to_string(image)
    return text

# PDF 파일을 열어서 extract_text_with_ocr 함수 실행 -> 벡터 데이터베이스에 저장하는 함수 작성
def load_and_retrieve_docs(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = extract_text_with_ocr(page)
                # 페이지에서 추출한 텍스트가 있을 때마다 text에 누적해서 저장
                if page_text:
                    text += page_text
    except Exception as e:
        return f"Error reading PDF file: {e}"
    
    # 만약 추출한 텍스트가 하나도 없는 경우 아래와 같은 메세지 출력하고 함수 종료
    if not text:
        return "No text found in the PDF file."
    
    # 추출한 텍스트가 있는 경우 
    docs = [Document(page_content=text)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()

# Document 리스트 -> 하나의 string으로 변환 함수 작성 
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG chain 함수 작성
def rag_chain(file, question):
    retriever = load_and_retrieve_docs(file)
    if isinstance(retriever, str):  # 리턴받은 값이 string인 경우 에러를 의미하므로 함수 중단
        return retriever
    
    retrieved_docs = retriever.get_relevant_documents(question)  # 가장 관련성이 높은 문서를 벡터 저장소에서 검색, 리스트로 생성
    formatted_context = format_docs(retrieved_docs) # 하나의 string으로 변환
    formatted_prompt = f"Question: {question}\n\nContext: {formatted_context}"
    response = ollama.chat(model='llama3.1', 
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
    title="[LLAMA 3] RAG 검색 활용 챗봇 시스템",
    description="PDF파일을 업로드하고 질문을 입력하면 답변을 생성해 드립니다. (영어로!)"
)

# app 실행
iface.launch()

# %%
