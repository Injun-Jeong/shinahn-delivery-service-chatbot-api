# app/setup.py
import os
import logging
import logging.handlers
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter

def setup_logging():
    """애플리케이션 로거를 설정하고 반환합니다."""
    logger = logging.getLogger('my_service_logger')
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.handlers.TimedRotatingFileHandler(
        os.path.join(log_dir, 'service.log'), when='midnight', interval=1, backupCount=7, encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

# .env 파일 로드
load_dotenv(override=True)

# LLM 인스턴스 및 로거 초기화
# 다른 파일에서 이 인스턴스들을 import하여 사용합니다.
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,
    check_every_n_seconds=0.1,
    max_bucket_size=10,
)

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    rate_limiter=rate_limiter
)

logger = setup_logging()