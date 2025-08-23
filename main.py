import os
import logging
import logging.handlers

from datetime import datetime  

from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.memory import ChatMessageHistory
from langchain_core.rate_limiters import InMemoryRateLimiter

from agents.qna_agent import get_qna_agent


# --- 로거 설정 ---
# 1. 로거 생성
logger = logging.getLogger('my_service_logger')
logger.setLevel(logging.INFO) # INFO 레벨 이상의 로그만 기록

# 2. 로그가 중복 출력되지 않도록 핸들러가 이미 있는지 확인
if not logger.handlers:
    # 3. 로그 형식(Formatter) 정의
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 4. 로그를 파일에 기록하는 핸들러(Handler) 생성
    # 매일 자정에 새 로그 파일을 생성하고, 최대 7개까지 백업 파일을 유지합니다.
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.handlers.TimedRotatingFileHandler(
        os.path.join(log_dir, 'service.log'), when='midnight', interval=1, backupCount=7, encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    # 5. 로거에 핸들러 추가
    logger.addHandler(file_handler)

    # (선택 사항) 콘솔에도 로그를 출력하고 싶을 경우
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)


from dotenv import load_dotenv
load_dotenv(override=True)


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



# 인메모리 세션 스토어: {세션 ID: ChatMessageHistory 객체} 형태로 저장됩니다.
session_histories = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """세션 ID를 기반으로 대화 기록을 가져오거나 새로 생성합니다."""
    if session_id not in session_histories:
        logger.info(f"[{session_id}] 세로운 세션을 시작합니다: {session_id}")
        session_histories[session_id] = ChatMessageHistory()
    return session_histories[session_id]

def end_session(session_id: str):
    """세션 ID에 해당하는 대화 기록을 로그 파일로 저장하고 메모리에서 삭제합니다."""
    if session_id in session_histories:
        history = session_histories[session_id]
        
        # 1. 로그 디렉터리 생성
        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)
        
        # 2. 파일명 생성 (yyyyMMddHHmmSS_세션ID.log)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f"{timestamp}_{session_id}.log"
        file_path = os.path.join(log_dir, file_name)
        
        # 3. 로그 파일 작성
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"Session ID: {session_id}\n")
                f.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*40 + "\n\n")
                
                for message in history.messages:
                    # message.type에 따라 'USER' 또는 'AI'로 표시
                    role = "USER" if message.type == "human" else "AI"

                    # additional_kwargs에서 타임스탬프를 가져옵니다. 없는 경우 'N/A' 처리.
                    msg_timestamp = message.additional_kwargs.get("timestamp", "N/A")
                    
                    # 로그에 타임스탬프를 함께 기록합니다.
                    f.write(f"[{msg_timestamp}] [{role}]\n{message.content}\n\n")
            
            logger.info(f"[{session_id}] 대화 기록이 로그 파일로 저장되었습니다: {file_path}")
        
        except Exception as e:
            logger.info(f"[{session_id}] 로그 파일 저장 중 오류가 발생했습니다: {e}")

        # 4. 메모리에서 대화 기록 삭제
        logger.info(f"[{session_id}] 세션을 종료하고 대화 기록을 삭제합니다: {session_id}")
        del session_histories[session_id]
        return True
    
    else:
        logger.info(f"[{session_id}] 존재하지 않는 세션 ID입니다: {session_id}")
        return False


# --- 가드레일 체인 정의 ---
guardrail_prompt_template = """
본 서비스(이하 땡겨요)는 신한은행이 '고객과 가맹점 모두에게 이로운 상생 금융'을 목표로 출시한 배달 애플리케이션입니다.
기존 배달앱 시장의 높은 중개수수료와 광고비 부담을 낮춰 소상공인(가맹점주)의 실질적인 소득 증대를 돕고, 
동시에 소비자에게는 다양한 할인 혜택과 금융 상품 연계 이점을 제공하는 것이 특징입니다.

당신은 사용자 질문이 '배달 서비스 플랫폼'과 관련된 내용인지 판단하는 엄격한 심사위원입니다.
주어진 대화 기록은 대화의 전체적인 맥락을 파악하는 데 사용하되, **가장 중요한 것은 마지막 사용자 질문입니다.**
이전 대화 내용보다 **가장 마지막에 들어온 사용자 질문의 의도를 최우선으로 판단**하세요.
사용자의 질문을 보고 관련이 있다면 'PASS', 전혀 관련이 없다면 'FAIL'이라고만 답해주세요.

오직 'PASS' 또는 'FAIL' 둘 중 하나로만 답변해야 합니다.

[서비스 관련 주제 예시]
- '음식주문 및 배달문의', '권리침해 신고', '결제 방법' 등 땡겨요 서비스와 관련 있는 질문
[서비스와 관련 없는 주제 예시]
- 타서비스(배달의민족, 쿠팡이츠, 요기요 등) 질문 또는 땡겨요 서비스와 관련 없는 질문(날씨, 정치, 주식 등)
---
사용자 질문: {user_input}
---
판단:"""
guardrail_prompt = PromptTemplate.from_template(guardrail_prompt_template)

logger.info(guardrail_prompt)

guardrail_chain = guardrail_prompt | llm_gemini | StrOutputParser()


# --- 의도분류 체인 정의 ---
intent_classification_prompt_json = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        본 서비스(이하 땡겨요)는 신한은행이 '고객과 가맹점 모두에게 이로운 상생 금융'을 목표로 출시한 배달 애플리케이션입니다.
        기존 배달앱 시장의 높은 중개수수료와 광고비 부담을 낮춰 소상공인(가맹점주)의 실질적인 소득 증대를 돕고, 
        동시에 소비자에게는 다양한 할인 혜택과 금융 상품 연계 이점을 제공하는 것이 특징입니다.

        당신은 사용자의 질문 의도를 분석하는 전문 분류기입니다.
        주어진 대화 기록은 대화의 전체적인 맥락을 파악하는 데 사용하되, **가장 중요한 것은 마지막 사용자 질문입니다.**
        이전 대화 내용보다 **가장 마지막에 들어온 사용자 질문의 의도를 최우선으로 판단**하여 다음 의도 중 가장 적절한 하나로 분류해 주세요.

        [의도]
        - QNA: 땡겨요 서비스 관련 일반 질문(예: 진행 중인 이벤트 등)
        - AICC
          1. '음식주문 및 배달문의': 주문 현황, 주문 변경(배달 위치, 메뉴 등), 주문 취소, 결제 수단, 현금영수증, 오배송
          2. '이용방법 및 회원정보 문의': 회원가입/탈퇴, 비회원 주문, 푸시 알림, 비밀번호찾기
          
        답변은 반드시 아래의 JSON 형식이어야 하며, 다른 어떤 텍스트도 포함해서는 안 됩니다.
        ```json
        {{
          "intent": "QNA|AICC",
          "desc": "사용자 질문 요약 및 의도 상세",
          "sentiment": "POSITIVE|NEUTRAL|NEGATIVE"
        }}
        ```""",
    ),
    ("user", "{input}"),
])
orchestration_chain = intent_classification_prompt_json | llm_gemini | JsonOutputParser()


# --- QnA Agent 체인 정의 ---
qna_chain = get_qna_agent()



def run_orchestrator_with_guardrail(user_id: str, session_id: str, user_input: str):
    logger.info(f"[{session_id}] 👤 User: {user_input} (Session: {session_id})")
    history = get_session_history(session_id)
    
    #  가드레일 체인으로 서비스 관련성 검사
    guardrail_result = guardrail_chain.invoke({"user_input": user_input}).strip()
    logger.info(f"[{session_id}] 🛡️ Guardrail Check: {guardrail_result}")

    # [변경] 타임스탬프를 포함하여 사용자 메시지 객체를 생성합니다.
    user_message = HumanMessage(
        content=user_input,
        additional_kwargs={"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    )

    if "FAIL" in guardrail_result:
        # 관련 없는 질문이면, 정해진 답변을 하고 종료
        response = "죄송합니다. 저는 땡겨요 서비스 관련 질문에만 답변해 드릴 수 있어요. 무엇을 도와드릴까요?"
        intent = "UNKNOWN"

        # [update history - Human msg] 생성해 둔 사용자 메시지 추가
        history.add_message(user_message) 
        # [update history - AI msg] 타임스탬프를 포함하여 최종 AI 메시지 객체를 생성하고 추가합니다.
        ai_message = AIMessage(
            content=response,
            additional_kwargs={"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        )
        history.add_message(ai_message)
        

    else:
        """사용자 입력과 대화 기록을 받아 의도를 분류하는 함수"""
        history_window_size = 3
        recent_history_messages = history.messages[-(history_window_size * 2):]

        # 체인을 실행합니다. 대화 기록과 사용자 입력을 함께 전달합니다.
        intent_json = orchestration_chain.invoke({
            "chat_history": recent_history_messages,
            "input": user_input,
        })
        logger.info(intent_json)

        # intent_json는 이제 {"intent": "..."} 형태의 딕셔너리입니다.
        intent = intent_json.get("intent", "UNKNOWN")
        logger.info(f"[{session_id}] 🤖 Orchestrator's Intent (JSON): **{intent_json}**")

        # tobe: call agent
        if intent == 'QNA':
            logger.info("✅ Routing to QnA Agent...")
            qna_input = {
                "question": user_input,
                "desc": intent_json.get("desc", ""),
                "sentiment": intent_json.get("sentiment", "NEUTRAL")
            }

            # 실제 QnA 체인을 호출합니다.
            #response = qna_chain.invoke(qna_input)
            # 스트리밍 방식으로 답변 생성
            print("🤖 AI: ", end="", flush=True)
            response = ""
            for chunk in qna_chain.stream(qna_input):
                print(chunk, end="", flush=True)
                response += chunk
            print()

        elif intent == 'AICC':
            logger.info("✅ Routing to AICC Agent... (미연결)")
            response = "[AICC 에이전트의 답변이 여기에 표시됩니다]"

        else:
            response = "무슨 말씀이신지 잘 모르겠어요. 좀 더 자세히 설명해 주시겠어요?"
        print(f"[{session_id}] 🤖 AI response : {response}")
        logger.info(f"[{session_id}] 🤖 AI response : {response}")

        # [update history - Human msg] 생성해 둔 사용자 메시지 추가
        history.add_message(user_message)     
        # [update history - AI msg] 타임스탬프를 포함하여 최종 AI 메시지 객체를 생성하고 추가합니다.
        final_ai_message = AIMessage(
            content=response,
            additional_kwargs={"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        )
        history.add_message(final_ai_message)

    result = {"response": response, "guardrail_result": guardrail_result, "intent": intent} 
    logger.info(f"[{session_id}] 👤 User: {user_input} (Session: {session_id}) ||| RESULT ||| {result}")
    # 최종 결과도 JSON으로 반환
    return result


# --- 실행 함수 ---
if __name__ == "__main__":
    # 테스트를 위한 고정된 세션 ID
    user_id_for_test = "user-12345"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    session_id_for_test = f"{user_id_for_test}_{timestamp}"

    logger.info("🤖 Orchestrator 테스트를 시작합니다. 종료하려면 'exit'을 입력하세요.")
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            end_session(session_id_for_test)
            logger.info(f"[{session_id_for_test}]프로그램을 종료합니다.")
            break
        
        result = run_orchestrator_with_guardrail(user_id_for_test, session_id_for_test, user_query)
        logger.info(f"[{session_id_for_test}] Agent: {result['response']}\n")    