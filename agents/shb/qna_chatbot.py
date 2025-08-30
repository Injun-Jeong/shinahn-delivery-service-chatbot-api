import os
import re
import json
import hashlib
import glob
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path
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

# FAISS 캐시 관련 환경변수
INDEX_DIR = os.getenv("INDEX_DIR", "/home/injun/workspace/shinahn-delivery-service-chatbot/agents/shb/index/faiss")
REBUILD_INDEX = os.getenv("REBUILD_INDEX", "false").lower() == "true"
ALLOW_DESERIALIZE = os.getenv("ALLOW_DESERIALIZE", "true").lower() == "true"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

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


def compute_data_fingerprint(data_dir: str = "/home/injun/workspace/shinahn-delivery-service-chatbot/agents/shb/data") -> str:
    """데이터 지문(fingerprint) 계산"""
    data_path = Path(data_dir)
    if not data_path.exists():
        return ""
    
    # 모든 .md 파일 경로를 안정 정렬
    md_files = sorted(glob.glob(str(data_path / "**/*.md"), recursive=True))
    
    # 각 파일의 경로와 내용을 해시에 추가
    hasher = hashlib.sha256()
    for file_path in md_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 파일 경로와 내용을 함께 해시
                file_info = f"{file_path}:{content}"
                hasher.update(file_info.encode('utf-8'))
        except Exception as e:
            print(f"[경고] 파일 읽기 실패: {file_path} - {e}")
    
    return hasher.hexdigest()

def _load_manifest() -> Optional[Dict]:
    """매니페스트 파일 로드"""
    manifest_path = Path(INDEX_DIR) / "manifest.json"
    if not manifest_path.exists():
        return None
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[인덱스] 매니페스트 로드 실패: {e}")
        return None

def _save_manifest(manifest_data: Dict):
    """매니페스트 파일 저장"""
    index_path = Path(INDEX_DIR)
    index_path.mkdir(parents=True, exist_ok=True)
    
    manifest_path = index_path / "manifest.json"
    try:
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[인덱스] 매니페스트 저장 실패: {e}")

def _load_faiss_if_valid(embeddings, data_dir: str = "/home/injun/workspace/shinahn-delivery-service-chatbot/agents/shb/data") -> Optional[FAISS]:
    """유효한 FAISS 인덱스 로드"""
    index_path = Path(INDEX_DIR)
    if not index_path.exists():
        return None
    
    manifest = _load_manifest()
    if not manifest:
        return None
    
    # 현재 설정과 매니페스트 비교
    current_fingerprint = compute_data_fingerprint(data_dir)
    if manifest.get('fingerprint') != current_fingerprint:
        print("[인덱스] 데이터 변경 감지 → 재생성 필요")
        return None
    
    if manifest.get('embedding_model') != EMBEDDING_MODEL:
        print("[인덱스] 임베딩 모델 변경 → 재생성 필요")
        return None
    
    if manifest.get('chunk_size') != CHUNK_SIZE or manifest.get('chunk_overlap') != CHUNK_OVERLAP:
        print("[인덱스] 청크 파라미터 변경 → 재생성 필요")
        return None
    
    # FAISS 인덱스 로드 시도
    try:
        if not ALLOW_DESERIALIZE:
            print("[인덱스] ALLOW_DESERIALIZE=false로 설정되어 로드 건너뜀")
            return None
        
        vectorstore = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=ALLOW_DESERIALIZE)
        doc_count = manifest.get('doc_count', 0)
        print(f"[인덱스] 로드 완료: {INDEX_DIR} (docs={doc_count})")
        return vectorstore
    except Exception as e:
        print(f"[인덱스] 로드 실패: {e}")
        return None

def _build_and_save_vectorstore(data_dir: str, embeddings) -> FAISS:
    """벡터스토어 빌드 및 저장"""
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
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(markdown_data_docs)
        
        # FAISS 인덱스 생성
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # 인덱스 저장
        index_path = Path(INDEX_DIR)
        index_path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(index_path))
        
        # 매니페스트 저장
        manifest_data = {
            'fingerprint': compute_data_fingerprint(data_dir),
            'embedding_model': EMBEDDING_MODEL,
            'chunk_size': CHUNK_SIZE,
            'chunk_overlap': CHUNK_OVERLAP,
            'doc_count': len(chunks),
            'built_at': datetime.utcnow().isoformat() + 'Z'
        }
        _save_manifest(manifest_data)
        
        print(f"[인덱스] 재생성 및 저장 완료: {INDEX_DIR} (chunks={len(chunks)})")
        return vectorstore
        
    except Exception as e:
        raise RuntimeError(f"벡터스토어 생성 중 오류: {str(e)}")

def _ensure_vectorstore(data_dir: str = "/home/injun/workspace/shinahn-delivery-service-chatbot/agents/shb/data"):
    """벡터스토어 보장 함수 - 캐시 우선 로드"""
    global _vs
    if _vs is None:
        try:
            embeddings = OpenAIEmbeddings(api_key=api_key, model=EMBEDDING_MODEL)
            
            # 강제 재빌드 체크
            if REBUILD_INDEX:
                print("[인덱스] REBUILD_INDEX=true로 설정되어 강제 재생성")
                _vs = _build_and_save_vectorstore(data_dir, embeddings)
            else:
                # 로드 시도
                _vs = _load_faiss_if_valid(embeddings, data_dir)
                if _vs is None:
                    # 로드 실패 시 재빌드
                    _vs = _build_and_save_vectorstore(data_dir, embeddings)
                    
        except Exception as e:
            raise RuntimeError(f"벡터스토어 준비 중 오류: {str(e)}")
    return _vs

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
    kept = [doc for doc, score in docs_with_scores if score >= (1 - SCORE_THRESHOLD)]
    
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

# CLI 워밍업 기능 (선택)
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="땡겨요 챗봇 인덱스 준비")
    parser.add_argument("--data", default="data", help="데이터 디렉토리 경로")
    parser.add_argument("--rebuild", action="store_true", help="강제 재생성")
    
    args = parser.parse_args()
    
    # 환경변수 오버라이드
    if args.rebuild:
        os.environ["REBUILD_INDEX"] = "true"
    
    try:
        _ensure_vectorstore(args.data)
        print("✅ 인덱스 준비 완료")
    except Exception as e:
        print(f"❌ 인덱스 준비 실패: {e}")
        exit(1)