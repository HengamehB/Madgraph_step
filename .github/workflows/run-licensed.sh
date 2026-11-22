#!/usr/bin/env bash
set -euo pipefail
export PATH="/tmp/madgraph-ci-tools:$PATH"
if [[ -d /opt/madgraph-ci/venv ]]; then
  export PATH="/opt/madgraph-ci/venv/bin:$PATH"
  cd /opt/madgraph-ci/source
fi
# Wolfram's raw-kernel licensing arguments apply to both -run and -script.
export WOLFRAMINIT="-pwfile !cloudlm.wolfram.com -entitlement ${ENTITLEMENT:?Missing entitlement}"
unset ENTITLEMENT
args=(--mode licensed)
if [[ ${TRACE_ORIGINAL_FIELDS:-false} == true ]]; then
  args+=(--trace-original-fields)
fi
status=0
python ci/run_benchmark.py "${args[@]}" || status=$?
python - <<'PY'
import json
import os
from pathlib import Path
directory = Path(os.environ["BENCHMARK_ARTIFACTS"])
entitlement = os.environ["WOLFRAMINIT"].split()[-1].encode()
if not (directory / "benchmark-licensed.json").is_file():
    raise SystemExit("Missing benchmark summary.")
for name in ("benchmark-licensed.json", "field-trace.json", "ranking.csv"):
    path = directory / name
    if path.is_file():
        with path.open("rb") as stream:
            overlap = b""
            for chunk in iter(lambda: stream.read(65536), b""):
                block = overlap + chunk
                if entitlement in block:
                    raise SystemExit("A benchmark artifact contains license information.")
                overlap = block[-(len(entitlement) - 1):]
(directory / ".safe-to-publish").write_text("checked\n", encoding="ascii")
summary = json.loads((directory / "benchmark-licensed.json").read_text())
for key in ("failure", "ufo_vertices", "width_comparisons", "production_comparison", "ranking"):
    if key in summary:
        print("Benchmark result", key, json.dumps(summary[key], ensure_ascii=True))
for command in summary.get("commands", []):
    if command.get("failure"):
        print("Failed benchmark command", command["action"], json.dumps(command["failure"], ensure_ascii=True))
trace_file = directory / "field-trace.json"
if trace_file.is_file():
    trace = json.loads(trace_file.read_text())
    for revision in ("original", "candidate"):
        result = trace.get(revision, {})
        print("UFO trace", revision, result.get("observation", result.get("status")))
        for key in ("kernel_events", "cli_failure", "reason", "reference_association", "trace_error"):
            if key in result:
                print("Trace detail", revision, key, json.dumps(result[key], ensure_ascii=True))
        fields = result.get("fields", {})
        if fields:
            print("Vertex summary", revision, json.dumps({k: v for k, v in fields.items() if k != "legs"}))
            for leg in fields["legs"]:
                print("Vertex leg", revision, json.dumps(leg, ensure_ascii=True))
PY
exit "$status"
