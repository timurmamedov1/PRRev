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
    @abstractmethod
    async def review(self, diff: str) -> ReviewResult:
        ...
