#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
VENV=""
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  if [[ "${args[i]}" == "--venv" && $((i+1)) -lt ${#args[@]} ]]; then
    VENV="${args[i+1]}"
  elif [[ "${args[i]}" == --venv=* ]]; then
    VENV="${args[i]#--venv=}"
  fi
done
[[ -n "$VENV" ]] || { echo "--venv is required" >&2; exit 64; }
VENV="$(cd -- "$VENV" && pwd -P)"
PYTHON="$VENV/bin/python"
[[ -x "$PYTHON" ]] || { echo "CI Python interpreter missing: $PYTHON" >&2; exit 66; }
[[ "$(stat -fc %T /sys/fs/cgroup)" == "cgroup2fs" ]] || { echo "cgroup v2 required" >&2; exit 70; }
command -v sudo >/dev/null
command -v systemd-run >/dev/null
sudo -n -v
WORKSPACE="${GITHUB_WORKSPACE:-$PWD}"
WORKSPACE="$(cd -- "$WORKSPACE" && pwd -P)"
unit="evidencedesk-gen-ci-$(date -u +%Y%m%dT%H%M%SZ)-$$"

# This is a single transient service so its server and both Python child
# processes inherit the same 10 GiB/4 CPU cgroup. No GitHub tokens or secrets
# are forwarded into the service or model process.
exec sudo systemd-run --wait --pipe --collect --unit="$unit" \
  --uid="$(id -u)" --gid="$(id -g)" \
  --property="WorkingDirectory=$WORKSPACE" \
  --property=MemoryMax=10737418240 --property=MemorySwapMax=0 \
  --property=CPUQuota=400% --property=Nice=10 \
  --property=KillMode=control-group --property=TimeoutStopSec=20s \
  --property=RuntimeMaxSec=45min \
  --setenv="HOME=${HOME:-/home/runner}" --setenv=PATH=/usr/bin:/bin \
  --setenv=OMP_NUM_THREADS=4 --setenv=MKL_NUM_THREADS=4 \
  --setenv=OPENBLAS_NUM_THREADS=4 --setenv=NO_PROXY=127.0.0.1,localhost \
  --setenv=no_proxy=127.0.0.1,localhost -- \
  "$PYTHON" "$HERE/run_ci_pair.py" "${args[@]}"
