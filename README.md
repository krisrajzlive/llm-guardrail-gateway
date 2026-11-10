# llm-guardrail-gateway

LangChain agent middleware for **multi-model routing, failover, tracing and inline evals** on **Amazon Bedrock**
(no third-party API keys), wrapped in a small Lambda service and shipped through a CI/CD pipeline where **a live
eval gate must pass before anything is deployed**.

```
push to main -> secret-scan + lint/types/tests -> live eval gate (3 Bedrock models) -> sam deploy -> smoke test
                                                       |
                                              exit 1 below threshold => deploy job skipped
```

## What is in the box

| Piece | Where | Purpose |
|---|---|---|
| Model registry | `models.py` | `init_chat_model("bedrock_converse:...")` aliases: `claude` (Haiku 4.5), `nova` (Nova Lite), `llama` (Llama 3.1 8B), via `us.` inference profiles |
| `ModelRouterMiddleware` | `middleware/router.py` | Picks the model per request from `GatewayContext.model` |
| `ProviderFailoverMiddleware` | `middleware/failover.py` | On error, tries the other models (never the active one) |
| `TracingMiddleware` | `middleware/tracing.py` | One span per model attempt: model, latency, tokens, error. Pluggable sinks (stdout/CloudWatch, JSONL, memory). LangSmith works alongside via env vars |
| `InlineEvalMiddleware` | `middleware/evals.py` | Per-response checks (PII, secrets, canary/prompt-leak, length). Report-only or enforcing |
| Prompt management | `prompts.py`, `prompts/chat/`, `scripts/prompt_registry.py` | System and user-turn prompts live in **Bedrock Prompt Management**, not in code. Text is reviewed in git, synced by CI, and the Lambda is pinned to an immutable version |
| Eval harness | `evals/` | 12-case dataset with deterministic scorers, per-model score table, exit code is the gate |
| Lambda handler | `handler.py` | Function URL (`AWS_IAM` auth) or direct invoke; Bedrock access via the function's IAM role |

Middleware order is `route -> failover -> trace -> inline-eval`, so failed attempts are traced and inline
evals only see the final response.

All three aliases use the Bedrock Converse API, so auth is the AWS credential chain: an IAM role on Lambda and in
CI (GitHub OIDC), a profile locally. Override any model with `LLM_MODEL_<ALIAS>`.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest && ruff check . && mypy
python -m evals.run --offline --min-score 1.0   # deterministic, no AWS needed
```

Live run (needs AWS credentials with Bedrock access, e.g. `AWS_PROFILE=...`; see `.env.example`):

```python
from llm_guardrail_gateway import build_agent, ask
agent = build_agent()
for alias in ("claude", "nova", "llama"):
    print(alias, ask(agent, "Explain quantum computing in one sentence.", alias))
```

```bash
python -m evals.run --min-score 0.85            # all three models
```

## Prompts

There are no prompt strings in application code.

```
prompts/chat/system.txt   {{reference_code}}   --sync-->  Bedrock managed prompt (DRAFT)  --gate-->  numbered version N
prompts/chat/user.txt     {{question}}                      evals run against DRAFT                  Lambda pinned to N
```

- **Source of truth:** `prompts/chat/*.txt`, reviewed in pull requests like code.
- **Eval gate:** CI syncs the files to the managed prompt's DRAFT and runs the evals against *that*, so a prompt change
  is gated exactly like a code change.
- **Deploy:** only if the gate passes is DRAFT snapshotted as an immutable version (re-used if unchanged), and
  `sam deploy` pins `PROMPT_VERSION` on the function. A rejected prompt edit never reaches production.
- **Runtime:** the Lambda reads the pinned version once per cold start; `{"health": true}` reports the live version.
- **Tests and offline evals** read the same files directly (`PROMPT_SOURCE=file`), so no AWS is needed.

Eval cases (`evals/dataset.json`) and the workflow's smoke-test message are test fixtures, not application prompts.

## CI/CD

`.github/workflows/ci-cd.yml`

1. **secret-scan** (gitleaks) and **test** (ruff, mypy, pytest with fake models, offline eval harness) run on every PR and push.
2. **eval-gate** runs on pushes to `main` only, so AWS credentials never reach fork PRs. It runs the dataset against
   all three models and exits non-zero if any scores below `EVAL_MIN_SCORE` (default `0.85`).
3. **deploy** `needs` all of the above. It assumes a separate, deploy-only AWS role via GitHub OIDC (no stored AWS
   keys), runs `sam build/deploy`, then smoke-tests the function. The eval gate uses its own role, which can only
   invoke Bedrock models.

### How a push flows through the pipeline

| # | Job | Runs on | What it does | Fails when | Effect of failing |
|---|---|---|---|---|---|
| 1 | `secret-scan` | PR + push | gitleaks over full history | a secret is found | everything after is skipped |
| 2 | `test` | PR + push | ruff, mypy, pytest (fake models), offline eval harness | any check fails | everything after is skipped |
| 3 | `eval-gate` | push to `main` | assume the eval role (OIDC), sync prompt to DRAFT, run live evals on 3 models | any model scores below `EVAL_MIN_SCORE` | **deploy is skipped** |
| 4 | `deploy` | after 1-3 | assume the deploy role (OIDC), publish prompt version, `sam build/deploy`, smoke test | build, deploy or smoke test fails | stack update rolls back |

Jobs 3 and 4 are linked with `needs:`, so a failed or skipped dependency means the next job never starts. Pull
requests only run jobs 1-2 and never receive AWS credentials. To watch it: `gh run watch`, and read the eval table in
the run's job summary.

### One-time setup

```bash
# 1. AWS: OIDC provider + least-privilege deploy and eval roles + artifact bucket.
#    New repos use immutable OIDC subject claims; read the prefix from GitHub:
gh api repos/<owner>/llm-guardrail-gateway/actions/oidc/customization/sub --jq .sub_claim_prefix
aws cloudformation deploy --stack-name llm-guardrail-gateway-bootstrap   --template-file infra/bootstrap.yaml --capabilities CAPABILITY_NAMED_IAM   --parameter-overrides GitHubRepo=<owner>/llm-guardrail-gateway OidcSubjectPrefix=<sub_claim_prefix>

# 2. GitHub configuration (from the stack outputs). Role ARNs go in secrets, not variables,
#    so the account id is masked in public workflow logs.
gh variable set AWS_REGION --body us-east-1
gh variable set ARTIFACT_BUCKET --body <ArtifactBucketName>
gh secret set AWS_DEPLOY_ROLE_ARN --body <DeployRoleArn>
gh secret set AWS_EVAL_ROLE_ARN --body <EvalRoleArn>
```

There are no API keys to manage. Create a `production` environment in the repository settings (optionally with
required reviewers for a manual approval before deploy), then protect `main` and require the `test` check.

### Calling the deployed function

The Function URL uses IAM auth, so requests must be SigV4-signed (for example with `awscurl`), or invoke directly:

```bash
aws lambda invoke --function-name llm-guardrail-gateway --cli-binary-format raw-in-base64-out \
  --payload '{"prompt": "Reply with exactly the word: pong", "model": "nova"}' out.json
```

## Design notes

- The model ids are cross-region inference profiles; the account needs Bedrock model access for them in the chosen
  region (Claude Haiku 4.5, Nova Lite and Llama 3.1 8B were verified invocable).
- Evals are deterministic (substring, regex, JSON keys, word limits, forbidden content) plus the same guardrail
  checks used inline. No LLM-as-judge, so the gate is cheap and repeatable. Live output is still slightly
  non-deterministic, so the threshold leaves one miss of headroom.
- Small models are not equally robust: in live runs Llama 3.1 8B leaked the system-prompt canary on the injection
  case. The eval gate scores it (0.92), and the inline eval middleware blocks that response in production.
- Failover catches any exception from a model call. That keeps availability high but can also fail over on bugs;
  the trace shows the original error on the `failover` event.
- Dependencies are ~178 MB unzipped for x86_64, inside Lambda's 250 MB zip limit.
