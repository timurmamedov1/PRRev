from prrev.llm.base import LLMProvider, ReviewResult


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        self.model = model
        self.api_key = api_key

    async def review(self, diff: str) -> ReviewResult:
        raise NotImplementedError
