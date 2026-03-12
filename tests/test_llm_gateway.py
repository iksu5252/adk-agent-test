from llm_gateway import (
    AdkLlmGateway,
    Backend,
    GatewayError,
    JsonOutputParser,
    ValidationError,
    build_default_parser,
    require_type,
)


class FakeClient:
    def __init__(self, output: str):
        self.output = output
        self.calls = []

    def generate(self, *, messages, system_instruction=None):
        self.calls.append({"messages": messages, "system_instruction": system_instruction})
        return self.output


def test_internal_is_default_route():
    internal = FakeClient('{"intent":"chat","confidence":91,"answer":"ok"}')
    chatgpt = FakeClient('{"intent":"chat","confidence":10,"answer":"no"}')

    gateway = AdkLlmGateway(
        internal_client=internal,
        chatgpt_client=chatgpt,
        parser=build_default_parser(),
    )

    result = gateway.request([{"role": "user", "content": "hello"}])

    assert result["intent"] == "chat"
    assert len(internal.calls) == 1
    assert len(chatgpt.calls) == 0


def test_chatgpt_on_off_switch():
    internal = FakeClient('{"intent":"chat","confidence":70,"answer":"internal"}')
    chatgpt = FakeClient('{"intent":"search","confidence":88,"answer":"gpt"}')

    gateway = AdkLlmGateway(
        internal_client=internal,
        chatgpt_client=chatgpt,
        parser=build_default_parser(),
        chatgpt_enabled=False,
    )

    try:
        gateway.request([{"role": "user", "content": "x"}], backend=Backend.CHATGPT)
    except GatewayError as exc:
        assert "disabled" in str(exc)
    else:
        raise AssertionError("Expected GatewayError")

    gateway.set_chatgpt_enabled(True)
    parsed = gateway.request([{"role": "user", "content": "x"}], backend=Backend.CHATGPT)
    assert parsed["answer"] == "gpt"


def test_alias_and_value_parsing():
    parser = build_default_parser()

    parsed = parser.parse('{"intent":"command","conf":"55","result":"ok"}')

    assert parsed == {"intent": "command", "confidence": 55, "answer": "ok"}


def test_custom_parser_for_extensibility():
    parser = JsonOutputParser(
        required_keys=("topic",),
        rules={"topic": require_type(str), "tags": require_type(list)},
        allow_unknown=True,
    )
    internal = FakeClient('{"topic":"news","tags":["a"],"extra":true}')
    chatgpt = FakeClient('{"topic":"news","tags":["a"]}')

    def add_trace(messages):
        return [*messages, {"role": "system", "content": "trace"}]

    def add_status(parsed):
        parsed["status"] = "ok"
        return parsed

    gateway = AdkLlmGateway(
        internal_client=internal,
        chatgpt_client=chatgpt,
        parser=parser,
        pre_hooks=[add_trace],
        post_hooks=[add_status],
    )

    parsed = gateway.request([{"role": "user", "content": "hello"}])

    assert parsed["topic"] == "news"
    assert parsed["extra"] is True
    assert parsed["status"] == "ok"
    assert internal.calls[0]["messages"][-1]["content"] == "trace"


def test_validation_error_for_bad_intent():
    parser = build_default_parser()
    with_unsupported = '{"intent":"unknown","confidence":20,"answer":"x"}'

    try:
        parser.parse(with_unsupported)
    except ValidationError as exc:
        assert "invalid value" in str(exc)
    else:
        raise AssertionError("Expected ValidationError")
