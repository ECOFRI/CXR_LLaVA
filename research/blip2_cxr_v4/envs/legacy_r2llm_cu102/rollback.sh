#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} != --execute || -z ${2:-} ]]; then
  echo 'Dry only. To execute after separate authorization: rollback.sh --execute /storage/<isolated-prefix>'
  exit 2
fi
prefix=$(realpath -m "$2")
if [[ $prefix != /storage/* || $prefix == /storage || ! -f $prefix/.p3_legacy_marker ]] || [[ $(cat "$prefix/.p3_legacy_marker") != legacy-r2llm-cu102 ]]; then
  echo 'Refusing unmarked or out-of-scope prefix' >&2
  exit 2
fi
conda env remove --prefix "$prefix" --yes
