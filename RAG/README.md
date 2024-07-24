# RAG를 활용한 PDF 파일 검색 챗봇 시스템

이 프로젝트는 Retrieval-Augmented Generation (RAG) 기술을 사용하여 PDF 파일에서 정보를 검색하고 가져오는 챗봇 시스템을 구현합니다.
## 참고

- 블로그 게시물(설명 포함) : https://smartest-suri.tistory.com/57
- 문의 : niceonesuri@gmail.com

## 개요

이 시스템은 다음 라이브러리를 사용합니다:
- `pdfplumber`
- `pytesseract`
- `langchain`
- `ollama`
- `chromadb`
- `gradio`

### 프로젝트 워크플로우

1. **텍스트 추출**: `pdfplumber`와 `pytesseract`를 사용하여 PDF 파일에서 텍스트를 추출합니다.
2. **텍스트 임베딩**: 추출한 텍스트를 `langchain`을 사용하여 분할하고 임베딩하여 Chroma 벡터 데이터베이스에 저장합니다.
3. **컨텍스트 준비**: 질문에 답하기 위해 벡터 데이터베이스에서 관련 콘텐츠를 검색하여 컨텍스트를 형성합니다.
4. **응답 생성**: 제공된 컨텍스트를 기반으로 `ollama` 라이브러리의 LLaMA3 모델을 사용하여 응답을 생성합니다.
5. **GUI**: `gradio`를 통합하여 사용자 친화적인 인터페이스를 제공합니다.

## 설치

1. 새로운 콘다 환경을 만들고 활성화합니다:
    ```bash
    conda create -n ragchatbot python=3.11 -y
    conda activate ragchatbot
    ```
2. 필요한 라이브러리를 설치합니다:
    ```bash
    pip install pdfplumber pytesseract ollama gradio langchain
    ```

## 사용법

1. **스크립트 실행**: Gradio 인터페이스를 실행합니다:
    ```python
    import gradio as gr
    from your_script import rag_chain

    iface = gr.Interface(
        fn=rag_chain,
        inputs=["file", "text"],
        outputs="text",
        title="[LLAMA 3] RAG 검색 활용 챗봇 시스템",
        description="PDF파일을 업로드하고 질문을 입력하면 답변을 생성해 드립니다. (영어로!)"
    )

    iface.launch()
    ```
2. **PDF 업로드 및 질문 입력**: PDF 파일을 업로드하고 질문을 입력하여 응답을 받습니다.

## 예시

* PDF 문서를 사용하고 질문을 입력하면 관련 있는 답변을 받습니다.
  
![image.jpg1](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbzH8LL%2FbtsIMQ0jhZP%2F4kSAc1oNV2ZtqmsmcNi6Qk%2Fimg.png) |![image.jpg2](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FFp1hm%2FbtsILgFKWqs%2FIqpLbTfHPhohXGq1t0lNHk%2Fimg.png)
--- | --- | 
