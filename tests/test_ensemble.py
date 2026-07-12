# tests for the ensemble provider

import pytest

from prrev.llm.base import LLMProvider, ReviewItem, ReviewResult
from prrev.llm.ensemble import EnsembleProvider


def _item(tag):
    return ReviewItem(severity="warning", file=f"{tag}.py", line=1, summary=tag, explanation=tag)


class StubProvider(LLMProvider):
    def __init__(
        self,
        *,
        summary="ok",
        max_tokens=100_000,
        fail_review=False,
        fail_reconcile=False,
    ):
        self.max_input_tokens = max_tokens
        self._summary = summary
        self._fail_review = fail_review
        self._fail_reconcile = fail_reconcile
        self.review_calls = 0
        self.reconcile_calls = 0
        self.count_calls = 0

    async def review(self, diff: str) -> ReviewResult:
        self.review_calls += 1
        if self._fail_review:
            raise RuntimeError("review down")
        return ReviewResult(items=[_item(self._summary)], summary=self._summary)

    async def count_tokens(self, text: str) -> int:
        self.count_calls += 1
        return len(text)

    async def reconcile(self, reviews: list[ReviewResult]) -> ReviewResult:
        self.reconcile_calls += 1
        if self._fail_reconcile:
            raise RuntimeError("judge down")
        merged = " + ".join(r.summary for r in reviews)
        return ReviewResult(items=[], summary=f"merged {merged}")


class TestEnsemble:
    def test_needs_two_providers(self):
        with pytest.raises(ValueError, match="two providers"):
            EnsembleProvider([StubProvider()], judge=StubProvider())

    def test_window_is_the_strictest(self):
        a = StubProvider(max_tokens=800_000)
        b = StubProvider(max_tokens=110_000)
        ensemble = EnsembleProvider([a, b], judge=a)
        assert ensemble.max_input_tokens == 110_000

    async def test_count_tokens_uses_smallest_window_provider(self):
        a = StubProvider(max_tokens=800_000)
        b = StubProvider(max_tokens=110_000)
        ensemble = EnsembleProvider([a, b], judge=a)
        await ensemble.count_tokens("hello")
        assert b.count_calls == 1
        assert a.count_calls == 0

    async def test_both_reviews_reach_the_judge(self):
        a = StubProvider(summary="claude view")
        b = StubProvider(summary="gpt view")
        ensemble = EnsembleProvider([a, b], judge=a)

        result = await ensemble.review("diff")
        assert a.review_calls == 1
        assert b.review_calls == 1
        assert a.reconcile_calls == 1
        assert result.summary == "merged claude view + gpt view"

    async def test_one_provider_down_degrades_with_notice(self):
        a = StubProvider(summary="claude view")
        b = StubProvider(fail_review=True)
        ensemble = EnsembleProvider([a, b], judge=a)

        result = await ensemble.review("diff")
        assert a.reconcile_calls == 0
        assert result.summary == "claude view"
        notices = [i for i in result.items if i.notice]
        assert len(notices) == 1
        assert "single-model review" in notices[0].summary

    async def test_both_providers_down_raises(self):
        a = StubProvider(fail_review=True)
        b = StubProvider(fail_review=True)
        ensemble = EnsembleProvider([a, b], judge=a)

        with pytest.raises(RuntimeError, match="review down"):
            await ensemble.review("diff")

    async def test_judge_down_falls_back_to_primary_with_notice(self):
        a = StubProvider(summary="claude view")
        b = StubProvider(summary="gpt view")
        judge = StubProvider(fail_reconcile=True)
        ensemble = EnsembleProvider([a, b], judge=judge)

        result = await ensemble.review("diff")
        assert result.summary == "claude view"
        notices = [i for i in result.items if i.notice]
        assert len(notices) == 1
        assert "reconciliation failed" in notices[0].summary
