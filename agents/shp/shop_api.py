import json
from pydantic import BaseModel
from typing import Optional, List
# --- 데이터 구조 정의 (Pydantic 모델) ---


class OrderInfo(BaseModel):
    """주문 정보 구조"""
    user_id: str
    tel_no: str


class OrderInfoOutput(BaseModel):
    """출력으로 contact_urls 리스트만 포함합니다."""
    contact_urls: List[str] # 생성된 URL 리스트


# --- 샘플 DB 데이터 ---
SAMPLE_DB = [
    OrderInfo(user_id="user-abc-123", tel_no="01040150588")
]


def _get_order_by_user_id(user_id: str) -> Optional[OrderInfo]:
    """샘플 DB에서 user_id로 주문 정보를 조회합니다."""
    for order in SAMPLE_DB:
        if order.user_id == user_id:
            return order
    return None


def _generate_contact_urls(tel_no: str) -> List[str]:
    """조회된 전화번호로 연락처 URL 리스트를 생성합니다."""
    # 전화번호 앞에 +82를 붙이고 맨 앞 0을 제거
    formatted_tel = f"%2B82{tel_no[1:]}"  # 01040150588 -> %2B821040150588
    urls = [
        f"https://sendmessage-sh-9224.twil.io/send-sms?to={formatted_tel}&body=배달이 지연되고 있습니다 사장님!",
        f"https://sendmessage-sh-9224.twil.io/make-call?to={formatted_tel}"
    ]
    return urls


def get_order_contact_info(user_id: str) -> OrderInfoOutput:
    """
    사용자 ID로 주문 정보를 조회하고 연락처 URL 리스트를 반환하는 LangGraph tool입니다.
    Args:
        user_id: 조회할 사용자 ID
    Returns:
        OrderInfoOutput: 연락처 URL 리스트가 포함된 결과
    """
    try:
        # :별:️ 샘플 DB에서 주문 정보 조회
        order_info = _get_order_by_user_id(user_id)
        if not order_info:
            print(f"[경고] user_id '{user_id}'에 해당하는 주문 정보를 찾을 수 없습니다.")
            return OrderInfoOutput(contact_urls=[])
        print(f":흰색_확인_표시: 주문 정보 조회 성공: {order_info.user_id} -> {order_info.tel_no}")
        # :별:️ 조회된 전화번호로 URL 리스트 생성
        contact_urls = _generate_contact_urls(order_info.tel_no)
        return OrderInfoOutput(
            contact_urls=contact_urls
        )
    except Exception as e:
        print(f"[오류] 데이터 조회 실패: {e}")
        return OrderInfoOutput(contact_urls=[])
# -----------------------------------------------------
# :별:️ Tool 테스트 및 사용 예제
# -----------------------------------------------------
if __name__ == "__main__":
    print(":흰색_확인_표시: LangGraph Tool 초기화 완료!")
    print(f":흰색_확인_표시: 샘플 DB 로드 완료! 총 {len(SAMPLE_DB)}개 주문 데이터")
    print("\n" + "="*50 + "\n")
    # tool 직접 호출 테스트
    result = get_order_contact_info("user-abc-123")
    if result and result.contact_urls:
        print(":오른쪽_화살표: [Tool 호출 성공]")
        # contact_urls 리스트 출력
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=4))
    else:
        print(":x: [Tool 호출 실패]")
    print("\n" + "="*50)
    print(":메모: LangGraph에서 사용하는 방법:")
    print("1. tools = [get_order_contact_info]")
    print("2. model.bind_tools(tools)")
    print("3. 또는 StateGraph에서 tool로 등록")