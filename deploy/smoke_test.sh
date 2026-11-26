#!/usr/bin/env bash
# End-to-end check of a running stack: submit an image, follow it to the end,
# download the mesh. Used by CI after `docker compose --profile cpu up`.
#
#   bash deploy/smoke_test.sh [http://localhost:8000]
set -euo pipefail

API="${1:-http://localhost:8000}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# A 64x64 red square on a transparent background.
echo 'iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAAYElEQVR42u3SQQ0AIAwEwYKQ6sD/Cx0YARF9kMCMgCa3aQQAAADwm1Y9MDP3zQFjrdKG/vsHCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAggggAACCCCAAAIIAAAAAPCiA/tKBEBZT6U+AAAAAElFTkSuQmCC' \
  | base64 -d > "$TMP/cube.png"

echo "==> Waiting for $API/readyz"
for _ in $(seq 60); do
  curl -fs "$API/readyz" >/dev/null && break
  sleep 2
done
curl -fsS "$API/readyz"; echo

echo "==> Submitting a job"
JOB="$(curl -fsS -X POST "$API/v1/jobs" \
  -H "Idempotency-Key: smoke-$(date +%s)" \
  -F "image=@$TMP/cube.png;type=image/png" \
  -F remove_background=false)"
echo "$JOB"
ID="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])' <<<"$JOB")"

echo "==> Following job $ID"
for _ in $(seq 90); do
  STATUS="$(curl -fsS "$API/v1/jobs/$ID" | python3 -c 'import json,sys; j=json.load(sys.stdin); print(j["status"], j["stage"], j["progress"])')"
  echo "    $STATUS"
  case "$STATUS" in
    succeeded*) break ;;
    failed*) curl -fsS "$API/v1/jobs/$ID"; exit 1 ;;
  esac
  sleep 2
done
[[ "$STATUS" == succeeded* ]] || { echo "timed out"; exit 1; }

curl -fsS -o "$TMP/result.glb" "$API/v1/jobs/$ID/result"
head -c 4 "$TMP/result.glb" | grep -q glTF || { echo "result is not a GLB"; exit 1; }
echo "==> OK: $(wc -c <"$TMP/result.glb") byte mesh"
curl -fsS "$API/metrics" | grep -E '^jobs_submitted_total' || true
