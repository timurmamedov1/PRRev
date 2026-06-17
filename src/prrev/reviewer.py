# orchestrator, takes diff + provider, returns structured review

from prrev.llm.base import LLMProvider, ReviewResult


async def review_diff(provider: LLMProvider, diff: str) -> ReviewResult:
    raise NotImplementedError
