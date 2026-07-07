from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ReviewItem:
    severity: str  # "critical" | "warning" | "suggestion"
    file: str
    line: int | None  # new-file line number (right side of diff)
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

    @abstractmethod
    def count_tokens(self, text: str) -> int: ...
