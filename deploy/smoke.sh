#!/usr/bin/env bash
# P3 冒烟:两次共享前缀请求,第二次 reused_blocks >= 3(多 block 复用)。
# 需先 ./deploy/run-local.sh
set -euo pipefail

BASE="${LAKE_SMOKE_BASE:-http://127.0.0.1:8080}"
URL="${BASE}/v1/chat/completions"
# 24 个 A → tokenizeMock 得 3 个满块(block_size=8),与 user 后缀无关的公共前缀。
SYS='AAAAAAAAAAAAAAAAAAAAAAAA'

echo "==> wait router"
for ((i=1; i<=60; i++)); do
  if curl -sf "${BASE}/healthz" >/dev/null; then break; fi
  if [[ $i -eq 60 ]]; then echo "ERROR: router not ready"; exit 1; fi
  sleep 0.25
done

post() {
  local user="$1"
  curl -sS "$URL" \
    -H 'Content-Type: application/json' \
    -d "$(python3 - <<PY
import json
print(json.dumps({
  "model": "tiny-llm",
  "messages": [
    {"role": "system", "content": "$SYS"},
    {"role": "user", "content": "$user"},
  ],
  "max_tokens": 4,
}))
PY
)"
}

echo "==> Req A"
A="$(post 'hello-AAA')"
echo "$A" | python3 -m json.tool
RA="$(echo "$A" | python3 -c 'import sys,json; print(json.load(sys.stdin)["lake"]["reused_blocks"])')"
PA="$(echo "$A" | python3 -c 'import sys,json; print(json.load(sys.stdin)["lake"]["prefill_blocks"])')"
MA="$(echo "$A" | python3 -c 'import sys,json; print(json.load(sys.stdin)["lake"]["mode"])')"

echo "==> Req B (shared system prefix ≥3 blocks)"
B="$(post 'hello-BBB')"
echo "$B" | python3 -m json.tool
RB="$(echo "$B" | python3 -c 'import sys,json; print(json.load(sys.stdin)["lake"]["reused_blocks"])')"
PB="$(echo "$B" | python3 -c 'import sys,json; print(json.load(sys.stdin)["lake"]["prefill_blocks"])')"
MB="$(echo "$B" | python3 -c 'import sys,json; print(json.load(sys.stdin)["lake"]["mode"])')"

echo "Req A: reused=$RA prefill=$PA mode=$MA"
echo "Req B: reused=$RB prefill=$PB mode=$MB"

if [[ "$RA" -ne 0 ]]; then
  echo "FAIL: Req A should be cold (reused_blocks==0), got $RA"
  exit 1
fi
if [[ "$PA" -lt 3 ]]; then
  echo "FAIL: Req A should prefill >=3 blocks (system prefix), got $PA"
  exit 1
fi
if [[ "$RB" -lt 3 ]]; then
  echo "FAIL: Req B reused_blocks should be >=3 (shared system prefix), got $RB"
  exit 1
fi
if [[ "$MB" != "COLOCATED" ]]; then
  echo "FAIL: P3 mode must be COLOCATED (no L0/D-direct), got $MB"
  exit 1
fi
echo "OK: multi-block prefix reuse across languages"
