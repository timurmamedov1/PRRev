# anthropic provider, uses tool use for structured output so we
# dont have to parse json from raw text

import os

import anthropic

from prrev.llm.base import LLMProvider, ReviewItem, ReviewResult

# tool schema that forces the model to call submit_review
# with the exact shape we need
REVIEW_TOOL = {
    "name": "submit_review",
    "description": "Submit a structured code review.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "1-2 sentence overall assessment of the diff.",
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "severity": {
                            "type": "string",
                            "enum": ["critical", "warning", "suggestion"],
                        },
                        "file": {
                            "type": "string",
                            "description": "filepath from the diff header",
                        },
                        "line": {
                            "type": ["integer", "null"],
                            "description": "new-file line number, or null if not identifiable",
                        },
                        "summary": {
                            "type": "string",
                            "description": "one line description of the issue",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "1-3 sentence explanation",
                        },
                    },
                    "required": ["severity", "file", "line", "summary", "explanation"],
                },
            },
        },
        "required": ["summary", "items"],
    },
}

SYSTEM_PROMPT = (
    "You are a senior code reviewer. You will receive a unified diff. "
    "Review it for bugs, security issues, logic errors, performance problems, "
    "and style issues. Be concise, no filler. "
    "If the diff is clean, submit an empty items array with a positive summary. "
    "Use the submit_review tool to return your review."
)


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        self.model = model
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.AsyncAnthropic(api_key=key)

    async def review(self, diff: str) -> ReviewResult:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[REVIEW_TOOL],
            # force the model to use our tool
            tool_choice={"type": "tool", "name": "submit_review"},
            messages=[{"role": "user", "content": diff}],
        )

        # find the tool use block in the response
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_review":
                data = block.input
                items = [
                    ReviewItem(
                        severity=item["severity"],
                        file=item["file"],
                        line=item.get("line"),
                        summary=item["summary"],
                        explanation=item["explanation"],
                    )
                    for item in data.get("items", [])
                ]
                return ReviewResult(items=items, summary=data.get("summary", ""))

        raise RuntimeError("model did not call submit_review tool")
