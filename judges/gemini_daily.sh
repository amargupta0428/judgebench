#!/bin/bash
# Daily-quota-aware Gemini runner (tier limit: 1,000 req/model/day, resets 07:00 UTC).
# Each cycle: sleep to just past reset, clear error cache entries, run until the
# wall. Exits when every test item is cleanly cached.
cd "$(dirname "$0")/.."
while true; do
  now=$(date -u +%s)
  target=$(date -u -v7H -v15M -v0S +%s)
  [ "$target" -le "$now" ] && target=$((target+86400))
  echo "$(date) sleeping $((target-now))s until quota reset"
  sleep $((target-now))
  .venv/bin/python - << 'PY'
import json, glob, os
n=0
for f in glob.glob('judges/cache/j2_gemini/*.json'):
    try:
        if json.load(open(f)).get('error'): os.remove(f); n+=1
    except Exception: os.remove(f); n+=1
print('cleared', n, 'error entries', flush=True)
PY
  left=$(.venv/bin/python -c "
import json,glob,hashlib
cached={f.split('/')[-1][:-5] for f in glob.glob('judges/cache/j2_gemini/*.json')}
items=[json.loads(l)['item_id'] for l in open('eval/testset_index.jsonl')]
print(sum(1 for i in items if hashlib.sha1(i.encode()).hexdigest()[:16] not in cached))")
  echo "$(date) remaining uncached: $left"
  if [ "$left" -eq 0 ]; then echo ALL_SCORED; break; fi
  JW=3 .venv/bin/python judges/j2_vlm.py run gemini
done
