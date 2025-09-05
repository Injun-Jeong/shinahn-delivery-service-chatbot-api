import os
import json
from typing import Dict, Any, List
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage
from langchain.tools import BaseTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# 환경 변수 로드
load_dotenv()

class CustomerQuery(BaseModel):
    """고객 질문 모델"""
    question: str
    user_id: str = None
    context: str = ""
    priority: str = "normal"  # low, normal, high, urgent

class SupportDecision(BaseModel):
    """지원 결정 모델"""
    needs_human_support: bool
    reason: str
    call_link: str = None
    urgency_level: str = "normal"  # low, normal, high, urgent
    question_category: str = None
    conversation_summary: str = None  # 상담사용 대화 요약

class KeywordAnalysisTool(BaseTool):
    """키워드 기반 분석 도구"""
    name: str = "keyword_analysis"
    description: str = "질문에서 키워드를 분석하여 긴급성과 카테고리를 판단합니다"
    
    def _run(self, query: str) -> str:
        """키워드 분석 실행"""
        # 키워드 정의
        sensitive_keywords = ["개인정보", "삭제", "해킹", "도용", "법적", "소송", "보상", "환불"]
        urgent_keywords = ["긴급", "즉시", "당장", "해킹", "도용", "사기", "금전적 손실", "로그인"]
        technical_keywords = ["설치", "설정", "오류", "버그", "업데이트", "호환성", "성능"]
        payment_keywords = ["결제", "환불", "취소", "요금", "청구", "인보이스", "영수증"]
        account_keywords = ["계정", "로그인", "회원가입", "비밀번호", "프로필", "권한"]
        
        # 키워드 매칭
        has_sensitive = any(keyword in query for keyword in sensitive_keywords)
        has_urgent = any(keyword in query for keyword in urgent_keywords)
        has_technical = any(keyword in query for keyword in technical_keywords)
        has_payment = any(keyword in query for keyword in payment_keywords)
        has_account = any(keyword in query for keyword in account_keywords)
        
        # 긴급성 점수 계산
        urgency_score = 3
        if has_urgent:
            urgency_score = 9
        elif has_sensitive:
            urgency_score = 7
        elif has_payment:
            urgency_score = 5
        elif has_account and ("안 되" in query or "문제" in query or "오류" in query):
            urgency_score = 6
        elif has_technical:
            urgency_score = 4
        
        # 긴급성 레벨
        if urgency_score >= 8:
            urgency_level = "urgent"
        elif urgency_score >= 6:
            urgency_level = "high"
        elif urgency_score >= 4:
            urgency_level = "normal"
        else:
            urgency_level = "low"
        
        # 카테고리 분류
        if has_urgent or (has_sensitive and urgency_score >= 7):
            category = "긴급/보안"
        elif has_technical and not has_urgent:
            category = "기술지원"
        elif has_payment and not has_urgent:
            category = "결제/환불"
        elif has_account and not has_urgent:
            category = "계정관리"
        elif has_sensitive and urgency_score < 7:
            category = "개인정보"
        else:
            category = "일반문의"
        
        return json.dumps({
            "urgency_score": urgency_score,
            "urgency_level": urgency_level,
            "category": category,
            "has_sensitive": has_sensitive,
            "has_urgent": has_urgent,
            "has_payment": has_payment,
            "has_account": has_account
        }, ensure_ascii=False)

class CustomerSupportAnalyzerAgent:
    """고객지원센터 질문 분석 에이전트 (LangChain 기반)"""
    
    def __init__(self):
        """에이전트 초기화"""
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # 도구 정의
        self.tools = [KeywordAnalysisTool()]
        
        # 프롬프트 템플릿
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 고객지원 질문 분석 전문가입니다.
            
고객의 질문을 분석하여 다음 정보를 제공해야 합니다:
1. 긴급성 점수 (1-10점)
2. 긴급성 레벨 (low/normal/high/urgent)
3. 질문 분류 (기술지원, 결제/환불, 계정관리, 개인정보, 일반문의, 긴급/보안)
4. 고객센터 연결 필요 여부 (true/false)
5. 분석 근거

사용 가능한 도구:
- keyword_analysis: 질문에서 키워드를 분석하여 긴급성과 카테고리를 판단

JSON 형태로 응답하세요:
{{
    "urgency_score": 점수,
    "urgency_level": "low/normal/high/urgent",
    "question_category": "분류명",
    "needs_human_support": true/false,
    "reasoning": "분류 근거"
}}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # 에이전트 생성
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # 에이전트 실행기
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def process_query(self, customer_query: CustomerQuery) -> SupportDecision:
        """고객 질문 처리 메인 함수"""
        
        # LangChain 에이전트로 분석
        analysis_result = self._analyze_with_langchain(customer_query)
        
        # SupportDecision 객체 생성
        decision = SupportDecision(
            needs_human_support=analysis_result["needs_human_support"],
            reason=analysis_result["reasoning"],
            urgency_level=analysis_result["urgency_level"],
            question_category=analysis_result["question_category"]
        )
        
        # 고객센터 연결이 필요한 경우 통화 링크와 요약 생성
        if decision.needs_human_support:
            decision.call_link = self._generate_call_link(customer_query, analysis_result)
            decision.conversation_summary = self._generate_conversation_summary(customer_query, analysis_result)
        
        return decision
    
    def _analyze_with_langchain(self, customer_query: CustomerQuery) -> Dict[str, Any]:
        """LangChain 에이전트를 사용한 분석"""
        
        # 분석 요청 메시지
        analysis_request = f"""
다음 고객 질문을 분석해주세요:

질문: {customer_query.question}
사용자 ID: {customer_query.user_id or 'anonymous'}
우선순위: {customer_query.priority}
추가 정보: {customer_query.context}

먼저 keyword_analysis 도구를 사용하여 키워드를 분석한 후, 전체적인 맥락을 고려하여 최종 판단을 내려주세요.
"""
        
        try:
            # 에이전트 실행
            result = self.agent_executor.invoke({
                "input": analysis_request,
                "chat_history": []
            })
            
            # 결과 파싱
            output = result["output"]
            
            # JSON 파싱 시도
            try:
                if isinstance(output, str):
                    # JSON 문자열에서 추출
                    if "{" in output and "}" in output:
                        start = output.find("{")
                        end = output.rfind("}") + 1
                        json_str = output[start:end]
                        parsed_result = json.loads(json_str)
                    else:
                        # 키워드 분석 결과만 있는 경우
                        parsed_result = json.loads(output)
                else:
                    parsed_result = output
                
                return {
                    "urgency_score": parsed_result.get("urgency_score", 3),
                    "urgency_level": parsed_result.get("urgency_level", "normal"),
                    "question_category": parsed_result.get("question_category", "일반문의"),
                    "needs_human_support": parsed_result.get("needs_human_support", False),
                    "reasoning": parsed_result.get("reasoning", "분석 완료")
                }
                
            except (json.JSONDecodeError, KeyError):
                # 파싱 실패 시 키워드 분석 도구 결과 사용
                keyword_result = self.tools[0]._run(customer_query.question)
                keyword_data = json.loads(keyword_result)
                
                needs_support = (
                    keyword_data.get("has_sensitive", False) or
                    keyword_data.get("has_urgent", False) or
                    keyword_data.get("has_payment", False) or
                    keyword_data.get("has_account", False)
                )
                
                return {
                    "urgency_score": keyword_data.get("urgency_score", 3),
                    "urgency_level": keyword_data.get("urgency_level", "normal"),
                    "question_category": keyword_data.get("category", "일반문의"),
                    "needs_human_support": needs_support,
                    "reasoning": f"키워드 분석 결과: 긴급성 {keyword_data.get('urgency_score', 3)}점, 분류: {keyword_data.get('category', '일반문의')}"
                }
                
        except Exception as e:
            # 에러 발생 시 키워드 분석 도구만 사용
            keyword_result = self.tools[0]._run(customer_query.question)
            keyword_data = json.loads(keyword_result)
            
            return {
                "urgency_score": keyword_data.get("urgency_score", 3),
                "urgency_level": keyword_data.get("urgency_level", "normal"),
                "question_category": keyword_data.get("category", "일반문의"),
                "needs_human_support": False,
                "reasoning": f"분석 중 오류 발생, 키워드 분석 결과 사용: {str(e)}"
            }
    
    def _generate_call_link(self, customer_query: CustomerQuery, analysis: Dict[str, Any]) -> str:
        """통화 링크 생성"""
        base_url = "https://support.example.com/call"
        user_id = customer_query.user_id or "anonymous"
        urgency_level = analysis["urgency_level"]
        category = analysis["question_category"]
        
        return f"{base_url}/{user_id}?urgency={urgency_level}&category={category}"
    
    def _generate_conversation_summary(self, customer_query: CustomerQuery, analysis: Dict[str, Any]) -> str:
        """상담사용 대화 요약 생성"""
        summary_template = f"""
고객 정보:
- 사용자 ID: {customer_query.user_id or 'anonymous'}
- 우선순위: {customer_query.priority}

질문 분석:
- 질문 내용: {customer_query.question}
- 질문 분류: {analysis['question_category']}
- 긴급성 레벨: {analysis['urgency_level']}
- 긴급성 점수: {analysis['urgency_score']}점

분석 근거: {analysis['reasoning']}

추가 정보: {customer_query.context if customer_query.context else '없음'}

상담 시 참고사항:
- 고객센터 연결이 필요한 사유: {analysis['reasoning']}
- 예상 상담 시간: {'즉시 처리 필요' if analysis['urgency_level'] == 'urgent' else '일반 상담'}
        """
        return summary_template.strip()

# 외부에서 호출할 수 있는 함수
def analyze_customer_support_query(question: str, user_id: str = None, priority: str = "normal", context: str = "") -> Dict[str, Any]:
    """
    고객지원 질문 분석 함수 (LangChain 기반)
    
    Args:
        question (str): 고객 질문
        user_id (str): 사용자 ID
        priority (str): 우선순위
        context (str): 추가 컨텍스트
    
    Returns:
        Dict[str, Any]: 분석 결과
    """
    agent = CustomerSupportAnalyzerAgent()
    query = CustomerQuery(
        question=question,
        user_id=user_id,
        priority=priority,
        context=context
    )
    
    result = agent.process_query(query)
    
    return {
        "needs_human_support": result.needs_human_support,
        "reason": result.reason,
        "call_link": result.call_link,
        "urgency_level": result.urgency_level,
        "question_category": result.question_category,
        "conversation_summary": result.conversation_summary
    }

# 테스트용 함수
def main():
    """테스트 실행 함수"""
    agent = CustomerSupportAnalyzerAgent()
    
    # 테스트 케이스들
    test_cases = [
        {
            "name": "일반 문의",
            "question": "제품 사용법을 알려주세요",
            "user_id": "user001",
            "priority": "normal"
        },
        {
            "name": "기술 지원",
            "question": "설치 중 오류가 발생했습니다",
            "user_id": "user002",
            "priority": "high"
        },
        {
            "name": "결제/환불",
            "question": "환불 정책에 대해 알고 싶습니다",
            "user_id": "user003",
            "priority": "normal"
        },
        {
            "name": "계정 관리",
            "question": "로그인이 안 되는데 어떻게 해야 하나요?",
            "user_id": "user004",
            "priority": "high"
        },
        {
            "name": "개인정보",
            "question": "개인정보 삭제 요청을 하고 싶습니다",
            "user_id": "user005",
            "priority": "high"
        },
        {
            "name": "긴급/보안",
            "question": "계정 해킹 의심이 있어서 긴급하게 도움이 필요합니다",
            "user_id": "user006",
            "priority": "urgent"
        }
    ]
    
    print("=== 고객지원센터 AI 분석기 테스트 (LangChain 기반) ===\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"📋 테스트 {i}: {test_case['name']}")
        print("-" * 50)
        
        query = CustomerQuery(
            question=test_case["question"],
            user_id=test_case["user_id"],
            priority=test_case["priority"]
        )
        
        result = agent.process_query(query)
        
        print(f"질문: {test_case['question']}")
        print(f"사용자 ID: {test_case['user_id']}")
        print(f"우선순위: {test_case['priority']}")
        print(f"질문 분류: {result.question_category}")
        print(f"긴급성 레벨: {result.urgency_level}")
        print(f"고객센터 연결 필요: {'예' if result.needs_human_support else '아니오'}")
        
        if result.needs_human_support:
            print(f"통화 링크: {result.call_link}")
            print(f"상담사용 요약:")
            print(result.conversation_summary)
        
        print(f"분석 근거: {result.reason}")
        print()

if __name__ == "__main__":
    main()
