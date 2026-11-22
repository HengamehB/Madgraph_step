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
python ci/run_benchmark.py --mode licensed
python - <<'PY'
import os
from pathlib import Path
report = Path(os.environ["BENCHMARK_ARTIFACTS"]) / "benchmark-licensed.json"
entitlement = os.environ["WOLFRAMINIT"].rsplit(" ", 1)[1].encode()
if any(entitlement in path.read_bytes() for path in (report, report.with_name("ranking.csv"))):
    raise SystemExit("The benchmark report contains license information.")
PY
