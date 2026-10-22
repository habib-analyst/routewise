#!/usr/bin/env bash
# RouteWise smoke test against a running stack (see README for startup).
# Shows, in order: a factual question routed to the small model, a code
# request routed to the large model, an economy-tier request whose long
# prompt gets truncated to the tier budget, and an over-limit request
# rejected with HTTP 400.
set -u
ROUTER=${ROUTER:-http://127.0.0.1:8100}

echo "== 1. factual, balanced tier (expect backend: small) =="
curl -s "$ROUTER/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "routewise-auto", "quality_tier": "balanced",
  "messages": [{"role": "user", "content": "What is the capital of France?"}]}' | head -c 600
echo; echo

echo "== 2. code, balanced tier (expect backend: large) =="
curl -s "$ROUTER/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "routewise-auto", "quality_tier": "balanced", "max_tokens": 32,
  "messages": [{"role": "user", "content": "Write a Python function that returns the nth Fibonacci number."}]}' | head -c 600
echo; echo

echo "== 3. long prompt, economy tier (expect input truncated to 256 tokens, output capped at 48) =="
LONG=$(python3 -c "print('Summarize the following text.\n\n' + 'The history of the steam engine spans many decades of incremental improvements by many inventors. ' * 60)")
curl -s "$ROUTER/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\": \"routewise-auto\", \"quality_tier\": \"economy\", \"messages\": [{\"role\": \"user\", \"content\": \"$LONG\"}]}" | head -c 600
echo; echo

echo "== 4. over-limit prompt, premium tier (expect HTTP 400) =="
HUGE=$(python3 -c "print('word ' * 6000)")
curl -s -w '\nHTTP %{http_code}\n' "$ROUTER/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\": \"routewise-auto\", \"quality_tier\": \"premium\", \"messages\": [{\"role\": \"user\", \"content\": \"$HUGE\"}]}" | head -c 400
echo

echo "== request log (budget vs actual, one line per request) =="
tail -n 4 "$(dirname "$0")/../logs/request_log.jsonl"
