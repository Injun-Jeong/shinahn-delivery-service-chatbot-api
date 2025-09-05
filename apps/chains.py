# app/chains.py
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from .setup import llm_gemini, logger # 설정 파일에서 LLM 인스턴스를 가져옵니다.
from agents.qna_agent import get_qna_agent

# --- 가드레일 체인 정의 ---
guardrail_prompt_template = """
본 서비스(이하 땡겨요)는 신한은행이 '고객과 가맹점 모두에게 이로운 상생 금융'을 목표로 출시한 배달 애플리케이션입니다.
기존 배달앱 시장의 높은 중개수수료와 광고비 부담을 낮춰 소상공인(가맹점주)의 실질적인 소득 증대를 돕고, 
동시에 소비자에게는 다양한 할인 혜택과 금융 상품 연계 이점을 제공하는 것이 특징입니다.

당신은 사용자 질문이 '배달 서비스 플랫폼'과 관련된 내용인지 판단하는 엄격한 심사위원입니다.
주어진 대화 기록은 대화의 전체적인 맥락을 파악하는 데 사용하되, **가장 중요한 것은 마지막 사용자 질문입니다.**
이전 대화 내용보다 **가장 마지막에 들어온 사용자 질문의 의도를 최우선으로 판단**하세요.
사용자의 질문을 보고 관련이 있다면 'PASS', 욕설 또는 부적절한 질문이라면 'FAIL'이라고만 답해주세요.

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
guardrail_chain = guardrail_prompt | llm_gemini | StrOutputParser()
logger.info("가드레일 체인이 생성되었습니다.")


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
        - CS: 회원정보 확인/변경, 강성민원, 보상 등 고객상담센터 직접 연결 필요 여부를 판단하여 연결 정보를 생성하는 에이전트
        - SHOP: 주문취소, 주문현황, 배달지연, 배달예정시간 변경, 현금영수증 발행 등 가맹점(식당)에 직접 연결이 필요한 경우 Shop API통해 전화 연결 기능과 문자 보내기 기능
          
        답변은 반드시 아래의 JSON 형식이어야 하며, 다른 어떤 텍스트도 포함해서는 안 됩니다.
        ```json
        {{
          "intent": "QNA|CS",
          "desc": "사용자 질문 요약 및 의도 상세",
          "sentiment": "POSITIVE|NEUTRAL|NEGATIVE"
        }}
        ```""",
    ),
    ("user", "{input}"),
])
orchestration_chain = intent_classification_prompt_json | llm_gemini | JsonOutputParser()
logger.info("의도분류 체인이 생성되었습니다.")

# --- QnA Agent 체인 정의 ---
qna_chain = get_qna_agent()
logger.info("QnA 에이전트 체인이 생성되었습니다.")