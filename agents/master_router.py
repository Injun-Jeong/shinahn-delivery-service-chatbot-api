from datetime import datetime
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from apps.session_manager import SessionManager
from apps.chains import guardrail_chain, orchestration_chain, qna_chain
from apps.setup import logger

from .shb import qna_chatbot
from .shs.cs_agnet_langchain import CustomerSupportAnalyzerAgent



class MasterRouter:
    def __init__(self):
        """
        마스터 라우터 에이전트를 초기화합니다.
        필요한 모든 구성요소(세션 관리자, 체인)를 여기서 설정합니다.
        """
        # 세션 관리 객체
        self.session_manager = SessionManager()
        
        # init agent
        self.guardrail_chain = guardrail_chain
        self.orchestration_chain = orchestration_chain
        self.qna_chain = qna_chain
        #self.cs_agent = CustomerSupportAnalyzerAgent()
        logger.info("🤖 Master Router Agent가 초기화되었습니다.")


    def handle_request(self, user_id: str, session_id: str, user_input: str):
        """사용자의 단일 요청을 처리하고 응답을 반환합니다."""
        logger.info(f"[{session_id}] 👤 User: {user_input}")
        history = self.session_manager.get_history(session_id)

        ############## 가드레일 agent 수행 ################        
        guardrail_result = self.guardrail_chain.invoke({"user_input": user_input}).strip()
        logger.info(f"[{session_id}] 🛡️ Guardrail Check: {guardrail_result}")

        user_message = HumanMessage(
            content=user_input,
            additional_kwargs={"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        )

        ############## 가드레일 수행값 확인(Fail인 경우, 가드레일을 통과하지 못함. Serving 종료) ################
        if "FAIL" in guardrail_result:
            response = "죄송합니다. 저는 땡겨요 서비스 관련 질문에만 답변해 드릴 수 있어요. 무엇을 도와드릴까요?"
            intent = "UNKNOWN"
            sentiment = "UNKNOWN"
            history.add_message(user_message)
            ai_message = AIMessage(content=response, additional_kwargs={"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
            history.add_message(ai_message)
        else: ############## 의도 분류 후, 서비스 agent 호출 ################
            # 컨텍스트(history) 관리
            recent_history = history.messages[-6:] # 최근 3턴

            # 의도 분류 
            intent_json = self.orchestration_chain.invoke({
                "chat_history": recent_history,
                "input": user_input,
            })
            intent = intent_json.get("intent", "UNKNOWN")
            sentiment = intent_json.get("sentiment", "UNKNOWN")
            logger.info(f"[{session_id}] 🤖 Intent: {intent_json}")

            if intent == 'QNA':
                logger.info(f"[{session_id}] ✅ Routing to QnA Agent...")
                


                """ shc version
                qna_input = { "question": user_input, **intent_json }
                response = ""
                print("🤖 AI: ", end="", flush=True)
                for chunk in self.qna_chain.stream(qna_input):
                    print(chunk, end="", flush=True)
                    response += chunk
                print()
                """

                """ shb version """

                param = f"""
                HUMAN은 사용자 질문입니다.
                중요한 것은 바로 이 HUMAN에 대해 답변을 잘 하는 것입니다.
                
                RECENT_HISTORY는 최근 대화 이력입니다.
                최근 대화 이력이 현재 질문에 대한 답변 생성에 너무 큰 영향을 끼치면 안 됩니다.
                현재 질문에 대한 답변 생성에 필요한 정보가 있다면 참고용으로만 활용하세요.

                [HUMAN: {user_input}, RECENT_HISTORY: {recent_history}]
                """

                print(f"💬 USER: {param}")
                response = qna_chatbot.answer(param)
                print(f"🤖 AI: {response}")


            elif intent == 'CS':
                logger.info(f"[{session_id}] ✅ Routing to CS Agent... (Not connected)")
                #response = "단순 Q&A가 아니군요. 새로운 Agent 개발이 필요합니다!"

                class CustomerQuery(BaseModel):
                    """고객 질문 모델"""
                    question: str
                    user_id: str = None
                    context: str = ""
                    priority: str = "normal"  # low, normal, high, urgent

                query = CustomerQuery(
                    question=user_input,
                    user_id=user_id,
                    priority="high"
                )
                print(f"💬 USER: {query}")
                cs_agent = CustomerSupportAnalyzerAgent()
                result = cs_agent.process_query(query)

                response = result.model_dump_json()
                
#                response = response + "\n\n  * 관련 문의는 고객센터 게시판에 전달되었으며, 추후 상담원을 통해 상세한 추가 답변 드리겠습니다."
                print(f"🤖 AI: {response}")

            
            else:
                response = "무슨 말씀이신지 잘 모르겠어요. 좀 더 자세히 설명해 주시겠어요?"

            logger.info(f"[{session_id}] 🤖 AI response: {response}")
            history.add_message(user_message)
            final_ai_message = AIMessage(content=response, additional_kwargs={"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
            history.add_message(final_ai_message)

        result = {"user_id": user_id, "session_id": session_id, "response": response, "sentiment": sentiment,"guardrail_result": guardrail_result, "intent": intent}
        logger.info(f"[{session_id}] 💬 Final result: {result}")
        return result

    