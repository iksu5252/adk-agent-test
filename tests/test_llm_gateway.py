from llm_gateway import (
    AdkLlmGateway,
    Backend,
    GatewayError,
    JsonOutputParser,
    ValidationError,
    build_default_parser,
    build_identity_parser,
    compose_system_instruction,
    require_type,
)


class FakeClient:
    def __init__(self, output: str):
        self.output = output
        self.calls = []

    def generate(self, *, messages, system_instruction=None):
        self.calls.append({"messages": messages, "system_instruction": system_instruction})
        return self.output


def test_default_routing_internal_backend():
    internal = FakeClient('{"intent":"chat","confidence":90,"answer":"ok"}')
    chatgpt = FakeClient('{"intent":"search","confidence":10,"answer":"no"}')

    gateway = AdkLlmGateway(internal_client=internal, chatgpt_client=chatgpt, parser=build_default_parser())

    result = gateway.request([{"role": "user", "content": "hello"}])

    assert result["intent"] == "chat"
    assert len(internal.calls) == 1
    assert len(chatgpt.calls) == 0


def test_chatgpt_toggle_on_off():
    internal = FakeClient('{"intent":"chat","confidence":90,"answer":"internal"}')
    chatgpt = FakeClient('{"intent":"chat","confidence":80,"answer":"gpt"}')
    gateway = AdkLlmGateway(
        internal_client=internal,
        chatgpt_client=chatgpt,
        parser=build_default_parser(),
        chatgpt_enabled=False,
    )

    try:
        gateway.request([{"role": "user", "content": "go"}], backend=Backend.CHATGPT)
    except GatewayError:
        pass
    else:
        raise AssertionError("Expected GatewayError")

    gateway.set_chatgpt_enabled(True)
    parsed = gateway.request([{"role": "user", "content": "go"}], backend=Backend.CHATGPT)
    assert parsed["answer"] == "gpt"


def test_description_is_injected_to_system_instruction():
    internal = FakeClient('{"intent":"chat","confidence":90,"answer":"ok"}')
    chatgpt = FakeClient('{"intent":"chat","confidence":80,"answer":"ok"}')

    gateway = AdkLlmGateway(
        internal_client=internal,
        chatgpt_client=chatgpt,
        parser=build_default_parser(),
        system_instruction="Return JSON only.",
        agent_description="Use ID/group schema strictly.",
    )

    gateway.request([{"role": "user", "content": "hello"}])

    sent = internal.calls[0]["system_instruction"]
    assert "Return JSON only." in sent
    assert "Agent description:" in sent
    assert "Use ID/group schema strictly." in sent


def test_compose_system_instruction_helper():
    assert compose_system_instruction(None, None) is None
    assert compose_system_instruction("a", None) == "a"
    assert "Agent description:" in compose_system_instruction(None, "b")


def test_identity_parser_fixes_merged_and_swapped_fields():
    parser = build_identity_parser()

    merged = parser.parse('{"ID":"ABCD22DD.BB_G", "group":"1", "age":24}')
    swapped = parser.parse('{"ID":"BB_G", "group":"ABCD22DD", "age":24}')

    assert merged == {"ID": "ABCD22DD", "group": "BB_G", "age": 24}
    assert swapped == {"ID": "ABCD22DD", "group": "BB_G", "age": 24}


def test_custom_parser_extensibility():
    parser = JsonOutputParser(
        required_keys=("topic",),
        rules={"topic": require_type(str)},
        allow_unknown=True,
    )

    parsed = parser.parse('{"topic":"news", "extra":true}')

    assert parsed["topic"] == "news"
    assert parsed["extra"] is True


def test_validation_error_for_invalid_intent():
    parser = build_default_parser()

    try:
        parser.parse('{"intent":"invalid","confidence":10,"answer":"x"}')
    except ValidationError:
        pass
    else:
        raise AssertionError("Expected ValidationError")
