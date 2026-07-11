# tests for llm providers using mocked clients

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prrev.llm.anthropic import REVIEW_TOOL, AnthropicProvider
from prrev.llm.openai import REVIEW_SCHEMA, OpenAIProvider


def _walk_schema_objects(schema):
    """yields every object-typed node in a json schema tree"""
    if isinstance(schema, dict):
        if schema.get("type") == "object" and "properties" in schema:
            yield schema
        for value in schema.values():
            yield from _walk_schema_objects(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from _walk_schema_objects(value)


SAMPLE_REVIEW = {
    "summary": "looks good overall",
    "items": [
        {
            "severity": "warning",
            "file": "app.py",
            "line": 10,
            "summary": "unused import",
            "explanation": "os is imported but never used",
        }
    ],
}


class TestAnthropicProvider:
    def test_missing_api_key_raises(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="ANTHROPIC_API_KEY not set"),
        ):
            AnthropicProvider(api_key=None)

    def test_accepts_explicit_key(self):
        provider = AnthropicProvider(api_key="sk-test-123")
        assert provider.client.api_key == "sk-test-123"

    @pytest.mark.asyncio
    async def test_parses_tool_use_response(self):
        provider = AnthropicProvider(api_key="sk-test-123")

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "submit_review"
        tool_block.input = SAMPLE_REVIEW

        mock_response = MagicMock()
        mock_response.content = [tool_block]
        mock_response.stop_reason = "tool_use"

        provider.client.messages.create = AsyncMock(return_value=mock_response)

        result = await provider.review("diff content")
        assert result.summary == "looks good overall"
        assert len(result.items) == 1
        assert result.items[0].severity == "warning"
        assert result.items[0].file == "app.py"
        assert result.items[0].line == 10

    @pytest.mark.asyncio
    async def test_malformed_item_skipped_with_notice(self):
        provider = AnthropicProvider(api_key="sk-test-123")

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "submit_review"
        tool_block.input = {
            "summary": "ok",
            "items": [
                {"severity": "warning"},  # missing required fields
                SAMPLE_REVIEW["items"][0],
            ],
        }

        mock_response = MagicMock()
        mock_response.content = [tool_block]
        mock_response.stop_reason = "tool_use"

        provider.client.messages.create = AsyncMock(return_value=mock_response)

        result = await provider.review("diff content")
        real = [i for i in result.items if not i.notice]
        notices = [i for i in result.items if i.notice]
        assert len(real) == 1
        assert len(notices) == 1
        assert "malformed" in notices[0].summary

    @pytest.mark.asyncio
    async def test_no_tool_call_raises(self):
        provider = AnthropicProvider(api_key="sk-test-123")

        text_block = MagicMock()
        text_block.type = "text"

        mock_response = MagicMock()
        mock_response.content = [text_block]
        mock_response.stop_reason = "end_turn"

        provider.client.messages.create = AsyncMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="model did not call submit_review"):
            await provider.review("diff content")

    @pytest.mark.asyncio
    async def test_truncated_response_raises(self):
        provider = AnthropicProvider(api_key="sk-test-123")

        mock_response = MagicMock()
        mock_response.content = []
        mock_response.stop_reason = "max_tokens"

        provider.client.messages.create = AsyncMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="stop_reason=max_tokens"):
            await provider.review("diff content")

    @pytest.mark.asyncio
    async def test_empty_items(self):
        provider = AnthropicProvider(api_key="sk-test-123")

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "submit_review"
        tool_block.input = {"summary": "clean code", "items": []}

        mock_response = MagicMock()
        mock_response.content = [tool_block]
        mock_response.stop_reason = "tool_use"

        provider.client.messages.create = AsyncMock(return_value=mock_response)

        result = await provider.review("diff content")
        assert len(result.items) == 0
        assert result.summary == "clean code"


class TestOpenAIProvider:
    def test_missing_api_key_raises(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="OPENAI_API_KEY not set"),
        ):
            OpenAIProvider(api_key=None)

    def test_accepts_explicit_key(self):
        provider = OpenAIProvider(api_key="sk-test-123")
        assert provider.client.api_key == "sk-test-123"

    @pytest.mark.asyncio
    async def test_parses_json_response(self):
        provider = OpenAIProvider(api_key="sk-test-123")

        mock_message = MagicMock()
        mock_message.content = json.dumps(SAMPLE_REVIEW)

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await provider.review("diff content")
        assert result.summary == "looks good overall"
        assert len(result.items) == 1
        assert result.items[0].severity == "warning"
        assert result.items[0].file == "app.py"

    @pytest.mark.asyncio
    async def test_truncated_response_raises(self):
        provider = OpenAIProvider(api_key="sk-test-123")

        mock_choice = MagicMock()
        mock_choice.finish_reason = "length"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="finish_reason=length"):
            await provider.review("diff content")

    @pytest.mark.asyncio
    async def test_empty_content_raises(self):
        provider = OpenAIProvider(api_key="sk-test-123")

        mock_message = MagicMock()
        mock_message.content = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="empty response"):
            await provider.review("diff content")

    @pytest.mark.asyncio
    async def test_empty_items(self):
        provider = OpenAIProvider(api_key="sk-test-123")

        mock_message = MagicMock()
        mock_message.content = json.dumps({"summary": "all good", "items": []})

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await provider.review("diff content")
        assert len(result.items) == 0
        assert result.summary == "all good"

    def test_count_tokens(self):
        provider = OpenAIProvider(api_key="sk-test-123")
        count = provider.count_tokens("hello world")
        assert isinstance(count, int)
        assert count > 0


class TestSchemaContract:
    # openai strict mode only accepts schemas where every object lists all its
    # property keys in required and sets additionalProperties to false

    def test_openai_schema_is_strict(self):
        assert REVIEW_SCHEMA["strict"] is True

    def test_openai_objects_require_all_properties(self):
        objects = list(_walk_schema_objects(REVIEW_SCHEMA["schema"]))
        assert objects, "expected at least one object node in the schema"
        for obj in objects:
            assert obj.get("additionalProperties") is False
            assert set(obj.get("required", [])) == set(obj["properties"])

    def test_anthropic_tool_is_strict(self):
        assert REVIEW_TOOL["strict"] is True

    def test_anthropic_objects_require_all_properties(self):
        objects = list(_walk_schema_objects(REVIEW_TOOL["input_schema"]))
        assert objects, "expected at least one object node in the schema"
        for obj in objects:
            assert obj.get("additionalProperties") is False
            assert set(obj.get("required", [])) == set(obj["properties"])
