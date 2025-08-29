# app/master_router.py
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from .session_manager import SessionManager
from .chains import guardrail_chain, orchestration_chain, qna_chain
from .setup import logger

class MasterRouter:
    def __init__(self):
        """
        마스터 라우터 에이전트를 초기화합니다.
        필요한 모든 구성요소(세션 관리자, 체인)를 여기서 설정합니다.
        """
        self.session_manager = SessionManager()
        self.guardrail_chain = guardrail_chain
        self.orchestration_chain = orchestration_chain
        self.qna_chain = qna_chain
        logger.info("🤖 Master Router Agent가 초기화되었습니다.")

    def handle_request(self, user_id: str, session_id: str, user_input: str):
        """사용자의 단일 요청을 처리하고 응답을 반환합니다."""
        logger.info(f"[{session_id}] 👤 User: {user_input}")
        history = self.session_manager.get_history(session_id)
        
        guardrail_result = self.guardrail_chain.invoke({"user_input": user_input}).strip()
        logger.info(f"[{session_id}] 🛡️ Guardrail Check: {guardrail_result}")

        user_message = HumanMessage(
            content=user_input,
            additional_kwargs={"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        )

        if "FAIL" in guardrail_result:
            response = "죄송합니다. 저는 땡겨요 서비스 관련 질문에만 답변해 드릴 수 있어요. 무엇을 도와드릴까요?"
            intent = "UNKNOWN"
            history.add_message(user_message)
            ai_message = AIMessage(content=response, additional_kwargs={"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
            history.add_message(ai_message)
        else:
            recent_history = history.messages[-6:] # 최근 3턴
            intent_json = self.orchestration_chain.invoke({
                "chat_history": recent_history,
                "input": user_input,
            })
            intent = intent_json.get("intent", "UNKNOWN")
            logger.info(f"[{session_id}] 🤖 Intent: {intent_json}")

            if intent == 'QNA':
                logger.info(f"[{session_id}] ✅ Routing to QnA Agent...")
                qna_input = { "question": user_input, **intent_json }
                response = ""
                print("🤖 AI: ", end="", flush=True)
                for chunk in self.qna_chain.stream(qna_input):
                    print(chunk, end="", flush=True)
                    response += chunk
                print()
            elif intent == 'AICC':
                logger.info(f"[{session_id}] ✅ Routing to AICC Agent... (Not connected)")
                response = "[AICC 에이전트의 답변이 여기에 표시됩니다]"
            else:
                response = "무슨 말씀이신지 잘 모르겠어요. 좀 더 자세히 설명해 주시겠어요?"

            logger.info(f"[{session_id}] 🤖 AI response: {response}")
            history.add_message(user_message)
            final_ai_message = AIMessage(content=response, additional_kwargs={"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
            history.add_message(final_ai_message)

        result = {"response": response, "guardrail_result": guardrail_result, "intent": intent}
        logger.info(f"[{session_id}] 💬 Final result: {result}")
        return result

    def start_cli_session(self):
        """테스트를 위한 대화형 CLI 세션을 시작합니다."""
        user_id = "cli_user"
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        session_id = f"{user_id}_{timestamp}"

        logger.info(f"🤖 마스터 라우터 테스트를 시작합니다. (Session: {session_id})")
        logger.info("종료하려면 'exit'을 입력하세요.")
        
        while True:
            try:
                user_query = input("You: ")
                if user_query.lower() == 'exit':
                    self.session_manager.end_session(session_id)
                    logger.info(f"[{session_id}] 프로그램을 종료합니다.")
                    break
                self.handle_request(user_id, session_id, user_query)
            except (KeyboardInterrupt, EOFError):
                self.session_manager.end_session(session_id)
                logger.info(f"\n[{session_id}] 강제 종료. 세션을 정리합니다.")
                break