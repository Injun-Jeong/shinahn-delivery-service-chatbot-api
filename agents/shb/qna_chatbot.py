import os
import re
from typing import Optional, List, Dict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "")

# 환경 변수 설정
K_DEFAULT = int(os.getenv("K_DEFAULT", "4"))
K_MAX = int(os.getenv("K_MAX", "8"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.3"))
DOC_CHAR_BUDGET = int(os.getenv("DOC_CHAR_BUDGET", "5000"))
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.5"))

_llm = None
_vs = None
_chain: Optional[RetrievalQA] = None

def _ensure_llm():
    global _llm
    if _llm is None:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 가 .env 에 설정되지 않았습니다.")
        _llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    return _llm

def pick_k(query: str, k_def: int, k_max: int) -> int:
    """동적으로 K 값을 결정"""
    q = query.lower()
    short = len(q) < 18
    easy_kw = any(w in q for w in ["환불","취소","영수증","쿠폰","포인트","계정"])
    hard_kw = any(w in q for w in ["지연","언제","늦","위치","배달","변경","기타"])
    
    k = k_def
    if easy_kw and short: 
        k = max(2, k_def - 1)
    if hard_kw or not short: 
        k = min(k_max, k_def + 2)
    
    return max(2, min(k, k_max))

def filter_docs_by_score_and_budget(docs_with_scores: List, k: int, fetch_k: int) -> List[Document]:
    """점수 임계치와 문자 예산으로 문서 필터링"""
    # 점수 기반 필터링 (FAISS는 낮은 점수가 더 유사함)
    kept = [doc for doc, score in docs_with_scores if score <= (1 - SCORE_THRESHOLD)]
    
    # 상위 k개 선택
    top = kept[:k]
    
    # 문자 예산 기반 필터링
    buf, chosen = 0, []
    for doc in top:
        if buf + len(doc.page_content) > DOC_CHAR_BUDGET:
            break
        chosen.append(doc)
        buf += len(doc.page_content)
    
    return chosen

def _parse_markdown_table(content: str) -> List[Dict]:
    """마크다운 테이블을 파싱하여 구조화된 데이터로 변환"""
    lines = content.strip().split('\n')
    if len(lines) < 3:
        return []
    
    # 헤더 추출
    header_line = lines[0]
    separator_line = lines[1]
    
    # 헤더에서 컬럼명 추출
    headers = [col.strip() for col in header_line.split('|')[1:-1]]
    
    # 데이터 행 파싱
    data_rows = []
    for line in lines[2:]:
        if line.strip() and not line.startswith('|:') and '|' in line:
            cols = [col.strip() for col in line.split('|')[1:-1]]
            if len(cols) == len(headers):
                row_data = dict(zip(headers, cols))
                data_rows.append(row_data)
    
    return data_rows

def _create_structured_documents(data_rows: List[Dict]) -> List[Document]:
    """구조화된 데이터를 LangChain Document로 변환"""
    documents = []
    
    for i, row in enumerate(data_rows):
        # 각 행을 구조화된 텍스트로 변환
        content = f"""
문의 종류: {row.get('inqry_tp_nm', '')}
문의 분류: {row.get('detl_tp_lcat_nm', '')} > {row.get('detl_tp_mcat_nm', '')} > {row.get('detl_tp_scat_nm', '')}
문의 제목: {row.get('ttle', '')}
문의 내용: {row.get('inqry_acpt_cont', '')}
답변 내용: {row.get('inqry_answ_cont', '')}
        """.strip()
        
        # 메타데이터 추가
        metadata = {
            'source': f'qna_data_{i}',
            'inqry_tp_nm': row.get('inqry_tp_nm', ''),
            'detl_tp_lcat_nm': row.get('detl_tp_lcat_nm', ''),
            'detl_tp_mcat_nm': row.get('detl_tp_mcat_nm', ''),
            'detl_tp_scat_nm': row.get('detl_tp_scat_nm', ''),
            'ttle': row.get('ttle', ''),
            'category': f"{row.get('detl_tp_lcat_nm', '')} > {row.get('detl_tp_mcat_nm', '')} > {row.get('detl_tp_scat_nm', '')}"
        }
        
        documents.append(Document(page_content=content, metadata=metadata))
    
    return documents

def _ensure_vectorstore(data_dir: str = "/home/injun/workspace/shinahn-delivery-service-chatbot/agents/shb/data"):
    global _vs
    if _vs is None:
        try:
            # 일반 마크다운 파일 로드
            loader = DirectoryLoader(data_dir, glob="**/*.md", loader_cls=TextLoader, recursive=True, loader_kwargs={"encoding": "utf-8"})
            docs = loader.load()
            
            # markdown_data.md 파일 특별 처리
            markdown_data_docs = []
            for doc in docs:
                if 'markdown_data.md' in doc.metadata.get('source', ''):
                    # 테이블 데이터 파싱
                    data_rows = _parse_markdown_table(doc.page_content)
                    if data_rows:
                        markdown_data_docs.extend(_create_structured_documents(data_rows))
                else:
                    # 일반 마크다운 파일은 그대로 사용
                    markdown_data_docs.append(doc)
            
            if not markdown_data_docs:
                raise RuntimeError(f"'{data_dir}' 폴더에서 .md 문서를 찾지 못했습니다.")
            
            # 청크 분할
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
            chunks = splitter.split_documents(markdown_data_docs)
            
            # 임베딩 및 벡터스토어 생성
            embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
            _vs = FAISS.from_documents(chunks, embeddings)
            
        except Exception as e:
            raise RuntimeError(f"벡터스토어 생성 중 오류: {str(e)}")
    return _vs

def build_chain(data_dir: str = "/home/injun/workspace/shinahn-delivery-service-chatbot/agents/shb/data"):
    """체인 구성 - LCEL 방식으로 변경"""
    global _chain
    llm = _ensure_llm()
    vs = _ensure_vectorstore(data_dir)
    
    # LCEL 체인 구성
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 땡겨요 배달앱의 전문 고객상담원입니다. 제공된 문서 근거를 우선 사용해 간결하고 정확히 답하세요."),
        ("human", "질문: {question}\n\n근거:\n{context}")
    ])
    
    def render_context(docs):
        return "\n\n---\n\n".join(doc.page_content[:1200] for doc in docs)
    
    def get_relevant_docs(query):
        # 동적 K 결정
        k = pick_k(query, K_DEFAULT, K_MAX)
        fetch_k = 4 * k
        
        # MMR 검색
        docs_with_scores = vs.similarity_search_with_score(query, k=fetch_k)
        
        # 필터링
        chosen_docs = filter_docs_by_score_and_budget(docs_with_scores, k, fetch_k)
        
        # 로그 출력
        print(f"[검색] K={k}, fetch_k={fetch_k}, 문서수={len(chosen_docs)}, 컷오프={len(docs_with_scores)-len(chosen_docs)}")
        
        return chosen_docs
    
    _chain = (
        {"question": RunnablePassthrough(), "context": lambda q: render_context(get_relevant_docs(q))} 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    return _chain

def answer(query: str) -> str:
    global _chain
    try:
        if _chain is None:
            build_chain()
        
        # LCEL 체인 직접 호출
        result = _chain.invoke(query)
        return result
    except Exception as e:
        return f"[오류] {e}"
