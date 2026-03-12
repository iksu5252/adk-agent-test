# Google ADK 기반 LLM Gateway

요청하신 내용(내부 서버 + ChatGPT on/off + JSON 파싱/검증 + 확장성)을 **Google ADK 스타일 오케스트레이션** 기준으로 정리한 스타터 코드입니다.

## 핵심 포인트

- 내부 모델 / ChatGPT 모델 라우팅 (`Backend.INTERNAL`, `Backend.CHATGPT`)
- ChatGPT 사용 on/off 토글 (`chatgpt_enabled`)
- JSON 응답 key alias 교정 + 타입/범위/허용값 검증
- Hook 기반 확장(`pre_hooks`, `post_hooks`)으로 추후 기능 추가 용이

## 구조

- `ModelClient` 프로토콜: ADK Agent/Runner 호출부를 감싸는 인터페이스
- `AdkLlmGateway`: 라우팅, 토글, 요청 실행, 파싱, hook 적용
- `JsonOutputParser`: JSON 파싱/정규화/검증
- Rule 유틸: `require_type`, `one_of`, `to_int`

## 빠른 사용 예시 (ADK 래핑 패턴)

```python
from llm_gateway import AdkLlmGateway, Backend, build_default_parser

# 실제 환경에서는 google.adk Agent/Runner 호출을 이 클래스에 래핑
class InternalAdkClient:
    def generate(self, *, messages, system_instruction=None) -> str:
        # TODO: 내부 ADK Agent 실행 후 JSON 문자열 반환
        return '{"intent":"chat","confidence":92,"answer":"안녕하세요"}'


class ChatGptClient:
    def generate(self, *, messages, system_instruction=None) -> str:
        # TODO: ChatGPT API 호출 후 JSON 문자열 반환
        return '{"intent":"search","confidence":85,"answer":"결과입니다"}'


gateway = AdkLlmGateway(
    internal_client=InternalAdkClient(),
    chatgpt_client=ChatGptClient(),
    parser=build_default_parser(),
    chatgpt_enabled=True,
    system_instruction="Always return JSON object.",
)

# 기본: internal
result = gateway.request([{"role": "user", "content": "요약해줘"}])

# 필요 시 ChatGPT 강제
result2 = gateway.request(
    [{"role": "user", "content": "검색해줘"}],
    backend=Backend.CHATGPT,
)
```

## 확장 아이디어

1. `ModelClient` 구현체를 ADK Agent별로 분리 (도메인 라우팅)
2. `JsonOutputParser.rules`에 도메인별 validator 추가
3. `pre_hooks/post_hooks`로 audit log, retry metadata, fallback 정책 연결
