from fastapi import FastAPI
from pydantic import BaseModel

from agents.master_router import MasterRouter
from apps.setup import logger

# 요청 본문의 데이터 구조를 정의하는 클래스
class Body(BaseModel):
    user_id: str
    session_id: str
    human: str | None = None

app = FastAPI()

router = MasterRouter()

# --- 기존 GET 엔드포인트 ---
@app.get("/")
def read_root():
    return {"Hello": "World"}

# --- 새로운 POST 엔드포인트 추가 ---
@app.post("/agent/")
def serving(body: Body): # 👈 파라미터로 Pydantic 모델을 받습니다.
    """
    LLM 애플리케이션의 메인 진입점(Entry Point).
    마스터 라우터를 초기화하고 CLI 세션을 시작합니다.
    """
    #logger.info("🚀 LLM Application starting...")
    # 마스터 라우터 인스턴스 생성
    result = router.handle_request(body.user_id, body.session_id, body.human)

    return result