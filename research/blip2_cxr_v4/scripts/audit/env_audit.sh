#!/usr/bin/env bash
set -u
CONDA=/home/student4/anaconda3/bin/conda
timeout 60 "$CONDA" env list
while IFS= read -r prefix; do
  [ -x "$prefix/bin/python" ] || continue
  echo "ENV $prefix"
  if ! timeout 60s "$CONDA" run -p "$prefix" python "$(dirname "$0")/env_probe.py" 2>&1; then
    echo "CONDA_RUN_FAILED; direct interpreter probe follows"
    timeout 60s "$prefix/bin/python" "$(dirname "$0")/env_probe.py" 2>&1
  fi
done < <(timeout 60 "$CONDA" env list | awk 'NF && $1 !~ /^#/ {print $NF}' | sort -u)
