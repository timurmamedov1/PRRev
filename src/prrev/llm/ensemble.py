# ensemble provider, reviews with every model and has a judge merge the results


import asyncio

from prrev.llm.base import LLMProvider, ReviewItem, ReviewResult


def _notice(summary: str, explanation: str) -> ReviewItem:
    return ReviewItem(
        severity="warning",
        file="ensemble",
        line=None,
        summary=summary,
        explanation=explanation,
        notice=True,
    )


class EnsembleProvider(LLMProvider):
    # runs the same review on every provider in parallel, then the judge
    # reconciles. degrades instead of failing: one provider down means a
    # single-model review with a notice, judge down means the primary
    # result with a notice

    def __init__(self, providers: list[LLMProvider], judge: LLMProvider):
        if len(providers) < 2:
            raise ValueError("ensemble needs at least two providers")
        self.providers = providers
        self.judge = judge
        # the strictest window wins so every sub-review fits everywhere
        self.max_input_tokens = min(p.max_input_tokens for p in providers)

    async def count_tokens(self, text: str) -> int:
        # count against the provider with the smallest window, keeps the
        # chunking decision conservative for the whole ensemble
        smallest = min(self.providers, key=lambda p: p.max_input_tokens)
        return await smallest.count_tokens(text)

    async def review(self, diff: str) -> ReviewResult:
        results = await asyncio.gather(
            *[p.review(diff) for p in self.providers],
            return_exceptions=True,
        )

        ok: list[ReviewResult] = []
        failures: list[Exception] = []
        for res in results:
            if isinstance(res, ReviewResult):
                ok.append(res)
            elif isinstance(res, Exception):
                failures.append(res)
            else:
                # cancellation and other BaseExceptions still propagate
                raise res

        if not ok:
            raise failures[0]

        if failures:
            result = ok[0]
            result.items.append(
                _notice(
                    f"{len(failures)} ensemble provider(s) failed, single-model review",
                    f"provider call failed: {failures[0]}",
                )
            )
            return result

        try:
            return await self.judge.reconcile(ok)
        except Exception as exc:
            primary = ok[0]
            primary.items.append(
                _notice(
                    "ensemble reconciliation failed, showing the primary review",
                    f"judge call failed: {exc}",
                )
            )
            return primary
