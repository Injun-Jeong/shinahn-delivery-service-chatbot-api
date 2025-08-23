from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.rate_limiters import InMemoryRateLimiter


from dotenv import load_dotenv
load_dotenv(override=True)

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,
    check_every_n_seconds=0.1, 
    max_bucket_size=10,  
)

llm_qna_agent = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    top_p=0.9,  # 누적 확률 90%까지
    top_k=40,   # 상위 40개 토큰만 고려
    streaming=True,  # 스트리밍 활성화
    rate_limiter=rate_limiter
)

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# --- 1. 인덱싱 기능: 문서를 임베딩하고 벡터 스토어를 생성하여 저장 ---
def _create_and_save_vector_store(knowledge_base_dir: str, vector_store_path: str, embedding_model: GoogleGenerativeAIEmbeddings):
    """
    지식 베이스 문서들을 로드, 분할, 임베딩하여 FAISS 벡터 스토어를 생성하고 디스크에 저장합니다.
    """
    print(f"'{knowledge_base_dir}' 경로의 문서로 벡터 스토어를 생성합니다...")
    
    # Load
    knowledge_base_path = Path(knowledge_base_dir)
    if not knowledge_base_path.exists():
        raise FileNotFoundError(f"지식 베이스 디렉터리를 찾을 수 없습니다: {knowledge_base_dir}")

    docs = []
    for file_path in knowledge_base_path.glob("*.txt"):
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs.extend(loader.load())

    if not docs:
        raise ValueError(f"지식 베이스 디렉터리에 로드할 문서가 없습니다: {knowledge_base_dir}")

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    # Store (Indexing)
    vector_store = FAISS.from_documents(split_docs, embedding_model)
    
    # Save to disk
    vector_store.save_local(vector_store_path)
    print(f"벡터 스토어를 '{vector_store_path}' 경로에 성공적으로 저장했습니다.")
    return vector_store




# --- 2. 검색/답변 기능: 저장된 벡터 스토어를 로드하여 RAG 체인 생성 ---
def get_qna_agent():

    knowledge_base_dir = "/Users/injun/Desktop/ai_project/llm-serve-module/res/qna"
    vector_store_path = "/Users/injun/Desktop/ai_project/llm-serve-module/vector_store/ddaenggyo_faq_index"

    """
    FAISS 벡터 스토어를 로드(없으면 생성)하여 RAG QnA 체인을 생성합니다.
    """
    vector_store_path_obj = Path(vector_store_path)
    
    if vector_store_path_obj.exists():
        # 벡터 스토어가 이미 존재하면 디스크에서 바로 로드
        print(f"'{vector_store_path}' 경로에서 기존 벡터 스토어를 로드합니다.")
        vector_store = FAISS.load_local(vector_store_path, embedding_model, allow_dangerous_deserialization=True)
    else:
        # 존재하지 않으면 새로 생성하고 저장
        print("기존 벡터 스토어가 없습니다. 새로 생성합니다.")
        vector_store = _create_and_save_vector_store(knowledge_base_dir, vector_store_path, embedding_model)

    # Retriever 생성
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}  # 상위 5개 문서만 검색
    )

    # 5. Prompt: RAG를 위한 프롬프트 템플릿을 정의합니다.

    rag_prompt = ChatPromptTemplate.from_template(
        """
        본 서비스(이하 땡겨요)는 신한은행이 '고객과 가맹점 모두에게 이로운 상생 금융'을 목표로 출시한 배달 애플리케이션입니다.
        기존 배달앱 시장의 높은 중개수수료와 광고비 부담을 낮춰 소상공인(가맹점주)의 실질적인 소득 증대를 돕고, 
        동시에 소비자에게는 다양한 할인 혜택과 금융 상품 연계 이점을 제공하는 것이 특징입니다.

        당신은 땡겨요 서비스 전문 상담원입니다. 
        주어진 컨텍스트 정보를 바탕으로 사용자 질문에 친절하고 명확하게 답변해 주세요.
        
        특히, 고객의 감정 상태가 '{sentiment}' 이므로, 이를 고려하여 어조를 신중하게 선택하세요.
        컨텍스트에 없는 내용이라면, "해당 내용은 제가 아는 정보에 없습니다. 고객센터에 문의해 주세요."라고 답해주세요.
    
        [컨텍스트]
        {context}
        
        [사용자 질문 및 요약]
        - 원본 질문: {question}
        - 질문 요약: {desc}
        """
    )

    # RAG 체인 구성
    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: retriever.invoke(x["question"])
        )
        | rag_prompt
        | llm_qna_agent
        | StrOutputParser()
    )

    return rag_chain

