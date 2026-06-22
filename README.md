# PRRev

A CLI tool that reviews code diffs using LLMs. Point it at a local repo or a GitHub PR URL. It sends the diff to Claude or GPT-4o and outputs a structured review with severity ratings, file references, and line numbers.

## Install

```bash
git clone https://github.com/timurmamedov1/PRRev.git
cd PRRev
pip install -e .
```

Requires Python 3.10+.

## Usage

```bash
# review uncommitted changes
prrev .

# review only staged changes
prrev . --staged

# review a specific commit
prrev . --commit abc123

# review a commit range
prrev . --range abc123..def456

# review a GitHub PR
prrev https://github.com/user/repo/pull/42

# post review as inline PR comments
prrev https://github.com/user/repo/pull/42 --post

# write review to a markdown file
prrev . --output review.md

# fail in CI if there are warnings or worse
prrev . --fail-on warning
```

## Configuration

Set API keys as environment variables:

```bash
export ANTHROPIC_API_KEY=sk-...
export OPENAI_API_KEY=sk-...
export GITHUB_TOKEN=ghp_...
```

Or use a config file at `~/.config/prrev/config.toml`:

```toml
[llm]
provider = "anthropic"
model = "claude-sonnet-4-20250514"
anthropic_api_key = "sk-..."

[github]
token = "ghp_..."

[review]
max_items = 20
```

You can also put a `.prrev.toml` in your repo root for per-project settings. API keys are ignored in repo config for security. They only load from env vars or the global config.

Precedence: CLI flags > env vars > repo config > global config > defaults.

## Providers

```bash
# use claude (default)
prrev . --provider anthropic

# use gpt-4o
prrev . --provider openai

# override the model
prrev . --provider anthropic --model claude-sonnet-4-20250514
```

## How It Works

The diff is sent to the LLM using each provider's structured output mechanism: Anthropic's tool use and OpenAI's `response_format` with a JSON schema. This guarantees the response matches the expected structure at the API level without fragile JSON text parsing.

For large diffs, PRRev automatically chunks by file based on actual token counts (not character estimates) and reviews each chunk in parallel. Results are merged and truncated by severity: suggestions are dropped first, then warnings. Critical issues are never dropped.

## Exit Codes

- `0:` review completed, no issues at or above `--fail-on` threshold
- `1:` issues found at or above `--fail-on` threshold
- `2:` tool error (missing API key, invalid args, network failure)

## Tech Stack

- **Typer:** CLI framework
- **Rich:** terminal output with colored severity panels
- **GitPython:** local diff extraction
- **httpx:** GitHub API (fetch PR diffs, post inline comments)
- **Anthropic SDK:** Claude API with tool use for structured output
- **OpenAI SDK:** GPT-4o API with structured output
- **tiktoken:** token counting for OpenAI
