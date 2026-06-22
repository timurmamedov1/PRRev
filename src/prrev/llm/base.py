from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ReviewItem:
    severity: str  # "critical" | "warning" | "suggestion"
    file: str
    line: int | None  # new-file line number (right side of diff)
    summary: str
    explanation: str


@dataclass
class ReviewResult:
    items: list[ReviewItem]
    summary: str


class LLMProvider(ABC):
    # max input tokens the model can handle, subclasses override
    max_input_tokens: int = 100_000

    @abstractmethod
    async def review(self, diff: str) -> ReviewResult:
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        ...
