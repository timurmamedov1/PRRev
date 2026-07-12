from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ReviewItem:
    severity: str  # "critical" | "warning" | "suggestion"
    file: str
    line: int | None  # new-file line for side=RIGHT, old-file line for side=LEFT
    summary: str
    explanation: str
    side: str = "RIGHT"  # RIGHT for additions, LEFT for deletions
    notice: bool = False  # tool-generated note (skipped file etc), not a model finding


@dataclass
class ReviewResult:
    items: list[ReviewItem]
    summary: str


class LLMProvider(ABC):
    # max input tokens the model can handle, subclasses override
    max_input_tokens: int = 100_000

    @abstractmethod
    async def review(self, diff: str) -> ReviewResult: ...

    # async bc counting can be a network call depending on the provider
    @abstractmethod
    async def count_tokens(self, text: str) -> int: ...

    async def reconcile(self, reviews: list[ReviewResult]) -> ReviewResult:
        # merge several reviews of the same diff into one, used by the
        # ensemble judge. providers that can judge override this
        raise NotImplementedError
