# llm-guardrail-gateway

LangChain agent middleware for **multi-provider routing, failover, tracing and inline evals**, wrapped in a
small Lambda service and shipped through a CI/CD pipeline where **a live eval gate must pass before anything
is deployed**.

```
push to main -> secret-scan + lint/types/tests -> live eval gate (3 providers) -> sam deploy -> smoke test
                                                       |
                                              exit 1 below threshold => deploy job skipped
```

## What is in the box

| Piece | Where | Purpose |
|---|---|---|
| Model registry | `models.py` | `init_chat_model` aliases: `openai:gpt-5.5`, `anthropic:claude-opus-4-8`, `google_genai:gemini-3.1-pro-preview` |
| `ModelRouterMiddleware` | `middleware/router.py` | Picks the provider per request from `GatewayContext.model` |
| `ProviderFailoverMiddleware` | `middleware/failover.py` | On error, tries the other providers (never the active one) |
| `TracingMiddleware` | `middleware/tracing.py` | One span per model attempt: model, latency, tokens, error. Pluggable sinks (stdout/CloudWatch, JSONL, memory). LangSmith works alongside via env vars |
| `InlineEvalMiddleware` | `middleware/evals.py` | Per-response checks (PII, secrets, canary/prompt-leak, length). Report-only or enforcing |
| Eval harness | `evals/` | 12-case dataset with deterministic scorers, per-model score table, exit code is the gate |
| Lambda handler | `handler.py` | Function URL (`AWS_IAM` auth) or direct invoke; keys read from Secrets Manager |

Middleware order is `route -> failover -> trace -> inline-eval`, so failed attempts are traced and inline
evals only see the final response.

> The Google provider id is `google_genai:` (underscore). `google-genai:` is not resolved by `init_chat_model`.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest && ruff check . && mypy
python -m evals.run --offline --min-score 1.0   # deterministic, no keys
```

Live run (keys in a git-ignored `.env`, see `.env.example`):

```python
from llm_guardrail_gateway import build_agent, ask
agent = build_agent()
for alias in ("openai", "anthropic", "google"):
    print(alias, ask(agent, "Explain quantum computing in one sentence.", alias))
```

```bash
python -m evals.run --min-score 0.85            # all three providers
```

## CI/CD

`.github/workflows/ci-cd.yml`

1. **secret-scan** (gitleaks) and **test** (ruff, mypy, pytest with fake models, offline eval harness) run on every PR and push.
2. **eval-gate** runs on pushes to `main` only, so provider keys never reach fork PRs. It runs the dataset against
   all three providers and exits non-zero if any scores below `EVAL_MIN_SCORE` (default `0.85`).
3. **deploy** `needs` all of the above. It assumes an AWS role via GitHub OIDC (no stored AWS keys), runs
   `sam build/deploy`, syncs provider keys from GitHub secrets into Secrets Manager, then smoke-tests the function.

### One-time setup

```bash
# 1. AWS: OIDC provider + least-privilege deploy role + artifact bucket
aws cloudformation deploy --stack-name llm-guardrail-gateway-bootstrap \
  --template-file infra/bootstrap.yaml --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides GitHubRepo=<owner>/llm-guardrail-gateway

# 2. GitHub repository variables
gh variable set AWS_REGION --body us-east-1
gh variable set AWS_DEPLOY_ROLE_ARN --body <DeployRoleArn output>
gh variable set ARTIFACT_BUCKET --body <ArtifactBucketName output>

# 3. GitHub repository secrets (prompted; values never appear on the command line)
gh secret set OPENAI_API_KEY && gh secret set ANTHROPIC_API_KEY && gh secret set GOOGLE_API_KEY
```

Create a `production` environment in the repository settings (optionally with required reviewers for a manual
approval before deploy). Then protect `main` and require the `test` check.

### Calling the deployed function

The Function URL uses IAM auth, so requests must be SigV4-signed (for example with `awscurl`), or invoke directly:

```bash
aws lambda invoke --function-name llm-guardrail-gateway --cli-binary-format raw-in-base64-out \
  --payload '{"prompt": "Reply with exactly the word: pong", "model": "openai"}' out.json
```

## Design notes

- Evals are deterministic (substring, regex, JSON keys, word limits, forbidden content) plus the same guardrail
  checks used inline. No LLM-as-judge, so the gate is cheap and repeatable. Live LLM output is still slightly
  non-deterministic, so the threshold leaves one miss of headroom.
- Failover catches any exception from a provider call. That keeps availability high but can also fail over on
  bugs; the trace shows the original error on the `failover` event.
- Dependencies are ~183 MB unzipped for x86_64, inside Lambda's 250 MB zip limit.
