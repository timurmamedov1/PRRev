# openai provider, uses response_format for structured output

import json
import os

import openai

from prrev.llm.base import LLMProvider, ReviewItem, ReviewResult

# json schema for structured output, same shape as the anthropic tool
REVIEW_SCHEMA = {
    "name": "review_response",
    "strict": True,
    "schema": {
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
                    "additionalProperties": False,
                },
            },
        },
        "required": ["summary", "items"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = (
    "You are a senior code reviewer. You will receive a unified diff. "
    "Review it for bugs, security issues, logic errors, performance problems, "
    "and style issues. Be concise, no filler. "
    "If the diff is clean, return an empty items array with a positive summary."
)


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        self.model = model
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = openai.AsyncOpenAI(api_key=key)

    async def review(self, diff: str) -> ReviewResult:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": diff},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": REVIEW_SCHEMA,
            },
        )

        data = json.loads(response.choices[0].message.content)
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
