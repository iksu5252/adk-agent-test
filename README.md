# Google ADK LLM Gateway (재작성)

처음부터 다시 정리한 버전입니다.

## 제공 기능

- 내부 LLM / ChatGPT 백엔드 라우팅
- ChatGPT on/off 제어
- `agent_description`을 system instruction에 합쳐서 매 요청 전달
- JSON 파싱 + 필드별 검증
- 깨진 데이터 자동 보정
  - `ID` + `group` 합쳐짐 (`ABCD22DD.BB_G`)
  - `ID`/`group` 값 서로 바뀜

## 핵심 객체

- `AdkLlmGateway`: 요청 라우팅 + 컨텍스트 주입 + 파싱 수행
- `ModelClient`: ADK Agent/Runner 래핑 인터페이스
- `JsonOutputParser`: alias/normalizer/rule 기반 검증기

## 빠른 예시

```python
from llm_gateway import AdkLlmGateway, Backend, build_identity_parser


class InternalClient:
    def generate(self, *, messages, system_instruction=None) -> str:
        return '{"ID":"ABCD22DD.BB_G", "group":"1", "age":24}'


class ChatGptClient:
    def generate(self, *, messages, system_instruction=None) -> str:
        return '{"ID":"BB_G", "group":"ABCD22DD", "age":24}'


gateway = AdkLlmGateway(
    internal_client=InternalClient(),
    chatgpt_client=ChatGptClient(),
    parser=build_identity_parser(),
    chatgpt_enabled=True,
    system_instruction="Always return JSON object only.",
    agent_description="ID/group/age 파싱. group은 PREFIX_CODE 포맷",
)

# 내부 모델
result1 = gateway.request([{"role": "user", "content": "파싱해줘"}])

# ChatGPT 강제
result2 = gateway.request(
    [{"role": "user", "content": "파싱해줘"}],
    backend=Backend.CHATGPT,
)
```
