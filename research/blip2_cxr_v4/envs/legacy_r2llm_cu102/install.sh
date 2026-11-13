#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} != --execute || -z ${2:-} ]]; then
  echo 'Dry only. To execute after separate authorization: install.sh --execute /storage/<isolated-prefix>'
  exit 2
fi
prefix=$(realpath -m "$2")
if [[ $prefix != /storage/* || $prefix == /storage || -e $prefix ]]; then
  echo 'Refusing prefix outside /storage or existing prefix' >&2
  exit 2
fi
export CONDA_PKGS_DIRS="$(dirname "$prefix")/.conda-pkgs"
export PIP_CACHE_DIR="$(dirname "$prefix")/.pip-cache"
export XDG_CACHE_HOME="$(dirname "$prefix")/.xdg-cache"
conda env create --prefix "$prefix" --file "$here/environment.yml"
printf '%s\n' 'legacy-r2llm-cu102' > "$prefix/.p3_legacy_marker"
"$prefix/bin/python" -m pip install -r "$here/requirements.lock.txt"
