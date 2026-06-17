from prrev.llm.base import LLMProvider, ReviewResult


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        self.model = model
        self.api_key = api_key

    async def review(self, diff: str) -> ReviewResult:
        raise NotImplementedError
