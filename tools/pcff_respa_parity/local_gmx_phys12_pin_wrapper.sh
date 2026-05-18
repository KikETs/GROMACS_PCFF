#!/usr/bin/env bash
set -euo pipefail

REAL_GMX="${GMX_BIN:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/build/bin/gmx}"
CPUSET="0-11"

args=("$@")
if [[ ${#args[@]} -gt 0 && "${args[0]}" == "mdrun" ]]; then
    for ((i = 0; i < ${#args[@]}; i++)); do
        if [[ "${args[i]}" == "-pin" && $((i + 1)) -lt ${#args[@]} ]]; then
            args[$((i + 1))]="on"
        fi
    done
fi

export OMP_PLACES=cores
export OMP_PROC_BIND=close

exec taskset -c "${CPUSET}" "${REAL_GMX}" "${args[@]}"
