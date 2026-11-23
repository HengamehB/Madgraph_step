#!/usr/bin/env bash
# Use the caller's activated Python environment and preserve their working directory.
set -euo pipefail
script_dir="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    for candidate in python python3.11 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            python_bin="$candidate"
            break
        fi
    done
fi
if [[ -z "$python_bin" ]] || ! command -v "$python_bin" >/dev/null 2>&1; then
    echo 'Python was not found. Activate your Python 3.11 environment (for example, conda activate madfieldrunner), or set PYTHON to its executable.' >&2
    exit 1
fi
exec "$python_bin" "$script_dir/pipeline_cli.py" "$@"
