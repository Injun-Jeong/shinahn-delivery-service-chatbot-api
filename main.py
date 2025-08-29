# main.py
from app.master_router import MasterRouter
from app.setup import logger # 로거를 여기서도 사용할 수 있습니다.

def main():
    """
    LLM 애플리케이션의 메인 진입점(Entry Point).
    마스터 라우터를 초기화하고 CLI 세션을 시작합니다.
    """
    logger.info("🚀 LLM Application starting...")
    
    # 마스터 라우터 인스턴스 생성
    router = MasterRouter()
    
    # 대화형 CLI 세션 시작
    router.start_cli_session()

    logger.info("👋 LLM Application finished.")

if __name__ == "__main__":
    main()