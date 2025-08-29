# app/session_manager.py
import os
from datetime import datetime
from langchain.memory import ChatMessageHistory
from .setup import logger  # 설정 파일에서 로거를 가져옵니다.

class SessionManager:
    def __init__(self):
        self.session_histories = {}

    def get_history(self, session_id: str) -> ChatMessageHistory:
        """세션 ID를 기반으로 대화 기록을 가져오거나 새로 생성합니다."""
        if session_id not in self.session_histories:
            logger.info(f"[{session_id}] 새로운 세션을 시작합니다.")
            self.session_histories[session_id] = ChatMessageHistory()
        return self.session_histories[session_id]

    def end_session(self, session_id: str) -> bool:
        """세션 ID에 해당하는 대화 기록을 로그 파일로 저장하고 메모리에서 삭제합니다."""
        if session_id in self.session_histories:
            history = self.session_histories[session_id]
            log_dir = "log"
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            file_name = f"{timestamp}_{session_id}.log"
            file_path = os.path.join(log_dir, file_name)

            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"Session ID: {session_id}\n")
                    f.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("="*40 + "\n\n")
                    for message in history.messages:
                        role = "USER" if message.type == "human" else "AI"
                        msg_timestamp = message.additional_kwargs.get("timestamp", "N/A")
                        f.write(f"[{msg_timestamp}] [{role}]\n{message.content}\n\n")
                logger.info(f"[{session_id}] 대화 기록이 로그 파일로 저장되었습니다: {file_path}")
            except Exception as e:
                logger.error(f"[{session_id}] 로그 파일 저장 중 오류 발생: {e}")

            logger.info(f"[{session_id}] 세션을 종료하고 대화 기록을 삭제합니다.")
            del self.session_histories[session_id]
            return True
        else:
            logger.warning(f"[{session_id}] 존재하지 않는 세션 ID입니다.")
            return False