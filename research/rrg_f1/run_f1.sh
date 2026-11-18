#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
umask 077
ulimit -c 0
export PYTHONDONTWRITEBYTECODE=1
exec python3 -m research.rrg_f1.supervisor
