# PRRev

[![tests](https://github.com/timurmamedov1/PRRev/actions/workflows/tests.yml/badge.svg)](https://github.com/timurmamedov1/PRRev/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/prrev)](https://pypi.org/project/prrev/)

A CLI tool that reviews code diffs using LLMs. Point it at a local repo or a GitHub PR URL. It sends the diff to Claude or GPT and outputs a structured review with severity ratings, file references, and line numbers.

## Install

```bash
pip install prrev
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

# review with both providers and reconcile the findings
prrev . --provider both

# show full tracebacks instead of short errors
prrev . --debug

# print the version
prrev --version
```

Note: brand new files don't appear in the diff until you `git add` them. PRRev will tell you when untracked files are being left out.

## Configuration

Set API keys as environment variables:

```bash
export ANTHROPIC_API_KEY=YOUR_ANTHROPIC_KEY
export OPENAI_API_KEY=YOUR_OPENAI_KEY
export GITHUB_TOKEN=YOUR_GITHUB_TOKEN
```

Or use a config file at `~/.config/prrev/config.toml`:

```toml
[llm]
provider = "anthropic"
model = "claude-haiku-4-5"          # used in single-provider mode
anthropic_model = "claude-opus-4-8" # per-provider, used in ensemble mode
openai_model = "gpt-5.4-mini"
anthropic_api_key = "YOUR_ANTHROPIC_KEY"

[github]
token = "YOUR_GITHUB_TOKEN"

[review]
max_items = 20
```

`model` is the single-provider default. For ensemble mode (`provider = "both"`), set `anthropic_model` and `openai_model` instead, `model` doesn't apply there.

You can also put a `.prrev.toml` in your repo root for per-project settings. API keys are ignored in repo config for security. They only load from env vars or the global config.

Precedence: CLI flags > env vars > repo config > global config > defaults.

`--post` needs a `GITHUB_TOKEN` that can write PR comments: a fine-grained personal access token (PAT) with **Pull requests: write**, or a classic PAT with the `repo` scope. A read-only token can fetch and review a PR but not post comments.

## Providers

```bash
# use claude (default)
prrev . --provider anthropic

# use openai
prrev . --provider openai

# override the model
prrev . --provider anthropic --model claude-opus-4-8
```

### Models you can use

**Anthropic:**

| Model | ID |
|---|---|
| Claude Fable 5 | `claude-fable-5` |
| Claude Sonnet 5 | `claude-sonnet-5` (default) |
| Claude Opus 4.8 | `claude-opus-4-8` |
| Claude Opus 4.7 | `claude-opus-4-7` |
| Claude Opus 4.6 | `claude-opus-4-6` |
| Claude Haiku 4.5 | `claude-haiku-4-5` |

**OpenAI:**

| Model | ID |
|---|---|
| GPT-5.6 Sol | `gpt-5.6-sol` (most capable) |
| GPT-5.6 Terra | `gpt-5.6-terra` (default, balanced) |
| GPT-5.6 Luna | `gpt-5.6-luna` (cheaper) |
| GPT-5.5 | `gpt-5.5` |
| GPT-5.4 mini | `gpt-5.4-mini` |
| GPT-5.4 nano | `gpt-5.4-nano` |

Older models like `gpt-4o` still work but are deprecated by OpenAI.

### Ensemble mode

Review with both providers at once and have one of them reconcile the results into a single review:

```bash
# both models review, claude reconciles (default judge)
prrev . --provider both

# pick the judge
prrev . --provider both --judge openai

# choose each provider's model
prrev . --provider both --model anthropic=claude-haiku-4-5 --model openai=gpt-5.4-mini
```

Findings flagged by both models are marked as higher confidence. If one provider fails, the review degrades to a single-model review with a notice instead of failing. Ensemble mode needs both API keys set and costs roughly double a single review, plus one reconciliation call per chunk.

## How It Works

The diff is sent to the LLM using each provider's structured output mechanism: Anthropic's tool use and OpenAI's `response_format` with a JSON schema. This guarantees the response matches the expected structure at the API level without fragile JSON text parsing.

For large diffs, PRRev automatically chunks by file based on actual token counts (not character estimates) and reviews each chunk in parallel. A single file too large for the context window is split further into hunk batches, so even oversized files get reviewed. Results are merged and truncated by severity: suggestions are dropped first, then warnings. Critical issues are never dropped.

## Exit Codes

- `0:` review completed, no issues at or above `--fail-on` threshold
- `1:` issues found at or above `--fail-on` threshold
- `2:` tool error (missing API key, invalid args, network failure)

Tool notices (a file skipped for size, a failed chunk) render as warnings but never count toward `--fail-on`, so CI fails only on real findings.

## Tech Stack

- **Typer:** CLI framework
- **Rich:** terminal output with colored severity panels
- **GitPython:** local diff extraction
- **httpx:** GitHub API (fetch PR diffs, post inline comments)
- **Anthropic SDK:** Claude API with tool use for structured output
- **OpenAI SDK:** GPT API with structured output
- **tiktoken:** token counting for OpenAI
