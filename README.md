# RouteWise: a cost, latency and quality model router with per-request token budgets

I am Habib Ur Rehman, an ML engineer. This is Project 4 of my AI
inference engineering programme. Project 1 (ServeCore) was about
serving, Project 2 (CacheLab) about the KV cache and speculative
decoding, Project 3 (QuantLab) about quantization tradeoffs. This one
is about the layer that sits in front of all of that: given two models
of different sizes, which one should answer each request, what token
budget should that request get, and what does the choice cost?

## The 30-second version

I built a router with an OpenAI-style API in front of two local models,
Qwen2.5-0.5B-Instruct (small) and Qwen2.5-1.5B-Instruct (large), both
in bf16 on a CPU-only VM (2 x86-64 vCPUs, 7.75 GiB RAM, no GPU). The
router classifies each request (factual, summary, code, reasoning),
scores its difficulty, assigns a token budget from the caller's
quality tier (economy, balanced, premium), enforces that budget, and
sends the request to the cheapest backend the policy says can handle
it. Every request is logged with its budget next to its actual usage.

I then ran one fixed 20-request workload three ways: everything to
the small model, everything to the large model, and routed. Headline
results, all measured on this machine:

| Strategy | Quality checks passed | Mean latency | p95 latency | Cost per 1000 requests |
|---|---|---|---|---|
| all to 0.5B | 17/20 | 21.2 s | 34.1 s | $0.50 |
| all to 1.5B | 15/20 | 33.2 s | 70.0 s | $0.78 |
| routed | 15/20 | 57.7 s | 74.2 s | $1.36 |

Read the cost column with the caveat in the unit-economics section:
the three strategies ran in different load windows on a shared VM, and
the routed run drew the worst window (one identical factual request
took 3.8 s in the all-small window and 335 s in the routed window).
The quality column has no such confound, and it carries the real
findings:

1. Routing matched the all-large quality (15/20) while sending 40% of
   requests to the small model. On this workload the router's split
   worked exactly as designed.
2. The small model beat both on quality (17/20). The difference is
   reasoning style, not knowledge: the quality checks ask whether the
   answer states its result inside the output budget, and the small
   model answers tersely while the large model spends its budget on
   preamble.
3. The binding constraint on reasoning was the token budget itself.
   Every reasoning request on both forced strategies consumed exactly
   its full 96-token output cap and was cut off mid-derivation. A
   budget is a cost control, but set it too tight and it amputates
   exactly the requests you routed up for quality.

## What I built

```
client (any OpenAI-style caller, or LiteLLM)
        |
        v
router/main.py        FastAPI, port 8100
  policy.py           classification, difficulty, budgets, routing
  logs/request_log.jsonl   one line per request: budget vs actual
        |
        +--> backend_server.py :8101   Qwen2.5-0.5B-Instruct (bf16)
        +--> backend_server.py :8102   Qwen2.5-1.5B-Instruct (bf16)
```

- `router/policy.py` is pure functions (task classification by keyword
  and length signals, a 0 to 1 difficulty score, per-tier budgets, the
  routing decision) and has unit tests in `router/test_policy.py` that
  run without any model.
- `router/main.py` exposes `POST /v1/chat/completions`, `GET
  /v1/models`, `GET /health` and `GET /stats`. The request body takes
  the usual `messages` and `max_tokens`, plus a `quality_tier` field
  (or an `X-Quality-Tier` header). Sending `model: "small"` or
  `model: "large"` forces a backend, which is how the benchmark's two
  forced strategies ran through the exact same code path as the routed
  one. The router loads only the shared Qwen2.5 tokenizer (both models
  use it), so its token counting and truncation match what the
  backends will see.
- `router/backend_server.py` serves one model per process behind the
  same completions endpoint and reports the backend's own generation
  time with every response, so model time can be separated from HTTP
  and queueing overhead.

### The routing policy

| Tier | Factual | Summary | Code | Reasoning |
|---|---|---|---|---|
| economy | small | small | small | small |
| balanced | small | small, large if prompt over 300 tokens | large | large |
| premium | large, small if prompt is 40 tokens or less | large | large | large |

In balanced, a difficulty score of 0.6 or more also escalates to the
large model. The reasoning is measurable, not vibes: in QuantLab the
0.5B model's perplexity was 11.70 against the 1.5B model's 4.16, and
its factual spot check passed 4/6 against 6/6, so tasks where being
wrong is expensive (code, reasoning) pay for the large model, while
short factual and summarisation work, where the small model is
demonstrably fine, do not.

One honest miss, recorded rather than hidden: the question "How many
sides does a hexagon have?" is classified as reasoning because it
contains the keyword "how many", and in the routed run it went to the
large model. It passed, and the cost of the miss is one slightly
slower factual answer, but keyword classifiers have edges like this
and a learned classifier (see the RouteLLM reference below) is the
proper fix.

### Token budgets and enforcement

| Tier | Max prompt tokens | Max output tokens |
|---|---|---|
| economy | 256 | 48 |
| balanced | 1024 | 96 |
| premium | 2048 | 128 |

Enforcement, all verified live in `results/smoke.json`:

- A prompt over its tier cap is truncated at token level (head 70%,
  tail 30% of the longest user message). The economy probe in the
  smoke test arrived with roughly 1900 tokens and was served with a
  250-token prompt and the action `input_truncated_to_tier_cap`
  recorded.
- Output is capped at the smaller of the client's request, the task
  default (factual 32, summary 64, code 96, reasoning 96) and the tier
  cap. The same probe's completion stopped at exactly 48 tokens, the
  economy cap.
- A prompt beyond the hard context limit of 4096 tokens is rejected
  with HTTP 400 before any backend is touched. The probe used 6030
  tokens and got the 400.
- Every request appends one line to `logs/request_log.jsonl` with the
  tier, task type, difficulty, budget, chosen backend, route reason,
  enforcement actions, and the actual prompt and completion tokens,
  so budget versus actual can be audited per request.

A design lesson the benchmark taught me: the premium tier's higher cap
(128) never helped its reasoning item, because the per-task default
(96) is what the client effectively requests and the grant is the
minimum of the two. A tier that is supposed to buy more reasoning room
has to scale the default, not just raise the ceiling. That is a
one-line policy change, and it is the first thing I would ship in a
version 2, now that I have measurements showing the cap, not the
model, was the constraint.

### Where LiteLLM fits, stated precisely

LiteLLM 1.106.2 is installed (pip --user) and used in two honest ways:

1. As a client. `bench/run_bench.py` sends three requests through
   `litellm.completion` in OpenAI-compatible mode against the router,
   proving that an existing LiteLLM user can put RouteWise in front of
   their models without changing their calling code. All three
   succeeded (`results/litellm_interop.json`). The first took 1446 s
   for reasons explained in the infrastructure section (an unpinned
   small backend faulting its pages back in); the other two took
   18.8 s and 22.5 s.
2. As an optional front proxy. `litellm_config.yaml` maps the router
   and the two backends as LiteLLM model entries for anyone who wants
   LiteLLM Proxy in the deployment.

What LiteLLM does not do here is the routing decision. Its Router does
load balancing, retries and fallbacks across deployments of a model
group; it does not classify requests by difficulty. The classifier,
the difficulty score and the budgets are my own policy code.

### Ollama and LM Studio status

Neither is installed on this VM, and I did not fake a result for
either. LM Studio has no headless Linux server I could install free
and quickly in this environment. Ollama could in principle be
installed, but it serves GGUF weights, and the models on this disk are
safetensors; covering the matrix properly would have meant a large
binary download plus a separate GGUF download of a model I already
had in another format, for a comparison of serving wrappers rather
than of routing. The gap is covered by the two paths that matter for
the routing question: direct HTTP backends, and the LiteLLM client
path above.

### Why the QuantLab variants are not live backends

QuantLab measured an INT8 0.5B at perplexity 11.72 (against 11.70 for
fp32) and an INT4 GPTQ variant at 13.71, which makes them sound like
ideal cheap backends. Their from-scratch decode rates were 0.45 and
0.32 tokens per second, an order of magnitude below the bf16 models,
because that implementation dequantizes the full weight matrix per
token. They informed this project's policy (the quality cost of going
smaller is small for easy tasks), but serving them live would have
measured the dequantizer, not the router.

## The benchmark

Workload (`bench/workload.py`): 20 requests, fixed across strategies.
Six short factual questions in the style of QuantLab's spot check,
four summarisation requests (one long, 370 input tokens), five code
requests and five reasoning requests. Each item carries the quality
tier a client would plausibly request (two economy, sixteen balanced,
two premium), and a mechanical quality check: the answer must contain
the fact, structure or result it should contain, and a summary must
also be at most 60% of its input's word count. Checks are identical
across strategies, so the pass rate is a fair quality proxy, not a
judge model's opinion.

Strategies: `model: "small"` (all to 0.5B), `model: "large"` (all to
1.5B), `model: "routewise-auto"` (the policy decides). Same items,
same tiers, same budgets; prompt tokens came out identical at 1647
for all three strategies, which is the integrity check that the
comparison is apples to apples. Decoding is greedy everywhere.
Results are appended per request to `results/<strategy>.jsonl` and
aggregated in `results/<strategy>_summary.json`.

### Full results

| | all to 0.5B | all to 1.5B | routed |
|---|---|---|---|
| Quality passed | 17/20 | 15/20 | 15/20 |
| factual (of 6) | 6 | 6 | 6 |
| summary (of 4) | 4 | 4 | 4 |
| code (of 5) | 5 | 4 | 4 |
| reasoning (of 5) | 2 | 1 | 1 |
| Mean latency | 21.2 s | 33.2 s | 57.7 s |
| Median latency | 24.0 s | 38.0 s | 49.0 s |
| p95 latency | 34.1 s | 70.0 s | 74.2 s |
| Prompt tokens | 1647 | 1647 | 1647 |
| Completion tokens | 1334 | 1209 | 1311 |
| Backend split | 20 small | 20 large | 8 small, 12 large |
| Measured wall time | 423.7 s | 663.7 s | 1153.1 s |

Per-category detail worth knowing:

- Code: the one failure on the large model (C3, a vowel counter) is a
  budget artefact. It used all 96 granted tokens and was cut off
  mid-function before reaching its `return`. The small model passed
  the same item with more compact code. Same cap, different verbosity,
  different outcome.
- Reasoning: R2 (a syllogism) is the cleanest illustration. The small
  model opens with "Yes, ..." and passes. The large model spends 96
  tokens on "To determine if every Bloops is a Lazzle ... we need to
  analyze these statements" and never lands the answer inside the
  budget. Both models likely "know" all five answers; the checks
  measure whether the answer is stated within budget, which is the
  quantity a budgeted production system actually cares about.
- Latency per item is dominated by output length, not model size. The
  large model answered the five short factual questions in 3 to 12 s
  each because it stops early; its terseness is also why its
  completion-token total is the lowest despite the slowest per-token
  rate.

## Unit economics

Cost model (`bench/cost_model.py`, output in
`results/cost_model.json`). Every input except the two labelled price
assumptions is measured above.

Assumptions, labelled:

- A1, instance price: $0.0848 per hour for an AWS c7i-flex.large
  (2 vCPU, 4 GiB), On-Demand, us-east-1, from the RunsOn EC2 finder
  listing fetched 2026-10-20. This VM (2 vCPUs, 7.75 GiB) is treated
  as equivalent to that instance class.
- A2, serving model: sequential single-request serving, no batching.
  A strategy costs the instance price multiplied by its measured wall
  time. Model load and idle time are excluded. In the routed
  deployment both backends share the one instance, so wall time, not
  per-backend time, is the cost driver.
- A3, GPU context only: an AWS g5.xlarge (one A10G) lists at about
  $1.006 per hour (Thunder Compute and Wring EC2 guides, October
  2026). I measured no GPU throughput, so I derive no GPU token cost;
  the figure is there to show the price gap a CPU router is operating
  under, nothing more.

| | all to 0.5B | all to 1.5B | routed |
|---|---|---|---|
| Cost of the 20-request workload | $0.0100 | $0.0156 | $0.0272 |
| Cost per 1000 requests | $0.50 | $0.78 | $1.36 |
| Cost per 1M total tokens | $3.35 | $5.47 | $9.18 |
| Cost per quality-passed request | $0.00059 | $0.00104 | $0.00181 |

Now the caveat, which is part of the result, not a footnote to hide
it in. These dollar figures inherit the load windows each strategy
happened to run in, and the windows were not equal: load average on
this shared VM ranged from about 4 to about 16 across the benchmark,
the host itself restarted more than once, and another tenant's
grading job held up to 3.7 GiB of RAM for hours. The routed run drew
the worst of it. The same factual request (F1) took 3.8 s in the
all-small window, 8.9 s in the all-large window and 335 s in the
routed window, when the small backend was running unpinned under
memory pressure. A solo micro-benchmark taken earlier the same day
(`results/micro_bench.json`) put the small model at 0.751 tokens per
second and the large at 0.297, while the warm pooled effective rates
across all benchmark rows came out at 1.77 and 1.86 tokens per second
respectively; the spread between those numbers is the load, not the
models.

So the honest reading of the economics is:

- Where routing saves money: against the all-large strategy at equal
  load, by construction. It spends large-model time only on the 12 of
  20 requests the policy escalates, and it matched all-large quality
  exactly. The mechanism is sound and the quality half of it is
  measured clean.
- Where it does not: against all-small on a workload like this one.
  When the small model passes 17 of 20 checks on its own, there is
  little quality left for routing to buy, and any large-model time is
  pure extra cost. Routing pays when the quality gap between the
  models is real for the traffic mix, which here it was only for code
  structure, and even there the budget interfered.
- The absolute dollars (fractions of a cent per workload) are a CPU
  artefact. At these throughputs the instance-hour price is spread
  over very few tokens; the same policy on a GPU backend at the
  g5.xlarge price point would be a different calculation, and I have
  not measured that calculation, so I am not presenting it.

## Infrastructure lessons, all measured on this VM

Building this on a shared, swapless, periodically restarting VM
taught me more about production inference than the routing did:

- Memory-mapped weights get evicted. Transformers loads safetensors
  by mmap, so the weights are file-backed pages. While the large
  backend sat idle, other tenants' cache pressure evicted them: its
  RSS fell to about 1.9 GB of its 3.1 GB, and a 32-token request then
  took 2780 s, most of it page faults (a solo run counted 8468 major
  faults in one 96-token generation). The fix is in
  `router/backend_server.py`: after loading, every tensor is cloned
  into anonymous memory, which a swapless kernel cannot reclaim. In
  the validation run (`results/pin_validation.json`), after 8 GB of
  deliberate cache churn plus an idle minute, the pinned backend held
  3404 MB resident and answered the same 32-token request in 47 s.
- The OOM killer is a tenant too. The kernel's oom_kill counter
  climbed repeatedly; the 3.4 GB backend is the largest single target
  on the box. `bench/ensure_stack.py` therefore health-checks the
  stack before every benchmark chunk, waits for a memory window
  (MemAvailable) before starting a backend, detects a child that died
  during startup, and retries. When the box genuinely could not hold
  both backends, the small one was started unpinned at half the RSS
  (`ROUTEWISE_PIN=0`), and for the final routed items the small
  backend was shut down entirely so the large one could fit.
- Everything long-lived dies; files survive. The host restarted at
  least three times during this build. Every benchmark result is
  appended to its JSONL the moment a request completes, and every
  chunk is resumable, which is the only reason a 60-request benchmark
  survived to completion here.
- Localhost is not local if the proxy says so. This VM's `no_proxy`
  contains bracketed IPv6 entries that crash httpx's URL parser, so
  the first stack appeared to hang: every health check was being sent
  to the egress proxy. The router and harness now sanitize `no_proxy`
  and use `trust_env=False` for backend calls, and all Hugging Face
  loads run with `HF_HUB_OFFLINE=1`, because a `from_pretrained`
  reachability check through this proxy can stall for minutes.

## Honest limits

- The quality checks are mechanical (keyword and structure matching),
  in the style of QuantLab's spot check. They are consistent and
  reproducible, and they measure "stated the right thing inside the
  budget", which is narrower than "the answer was good".
- Twenty requests is a small workload. It is enough to compare
  strategies on identical items, not enough to estimate production
  percentiles; the p95 of 20 requests is close to the max and I have
  reported the max alongside it in the summaries.
- All serving here is single-request and sequential. The strongest
  economic argument for a small model, its throughput under
  concurrency (ServeCore measured the serving side of that), is
  outside what this benchmark exercised.
- Latency and cost figures are entangled with this VM's load, as
  documented above. Quality figures are not.
- The classifier is keyword based. It has known edges (the hexagon
  case), and no learning behind it.

## Run it yourself

```bash
cd router
python3 backend_server.py --name small --port 8101 \
    --model-path /path/to/Qwen2.5-0.5B-Instruct
python3 backend_server.py --name large --port 8102 \
    --model-path /path/to/Qwen2.5-1.5B-Instruct
python3 main.py            # router on :8100
bash ../bench/smoke.sh     # the four probes from the smoke test
python3 ../router/test_policy.py        # policy unit tests, no models
python3 ../bench/run_bench.py           # full benchmark, quiet machine
python3 ../bench/run_bench.py --phase strategy --strategy routed --max-new 5
python3 ../bench/cost_model.py          # unit economics from results
python3 ../charts/make_charts.py        # charts from results
```

## References

- Ong et al., "RouteLLM: Learning to Route LLMs with Preference
  Data", arXiv:2406.18665, 2024. The learned version of what my
  keyword policy approximates.
- Chen, Zaharia, Zou, "FrugalGPT: How to Use Large Language Models
  While Reducing Cost and Improving Performance", arXiv:2305.05176,
  2023. The cascade and cost framing behind the tier design.
- LiteLLM documentation and Router (BerriAI): the client and proxy
  layer used for the interoperability test, and the reference point
  for what a production routing layer handles beyond classification
  (fallbacks, budgets per key, observability).
- My own QuantLab results (`../quantlab/results/summary.json`) for
  the per-model quality numbers the policy thresholds were set from.
