#!/usr/bin/env bash
set -euo pipefail
dry=0
[[ ${1-} == -n ]] && { dry=1; shift; }
dir=${1-}
[[ -d $dir ]] || { echo "no such dir: $dir" >&2; exit 2; }
export LC_ALL=C
declare -A taken=()
skipped=0
mapfile -d '' names < <(find "$dir" -mindepth 1 -maxdepth 1 -type f -printf '%f\0' | sort -z)
for b in "${names[@]}"; do
    lower=$(printf '%s' "$b" | tr 'A-Z' 'a-z')
    [[ $lower == "$b" ]] && continue
    if [[ -e $dir/$lower || -L $dir/$lower || -n ${taken[$lower]-} ]]; then
        echo "skip: $b (exists)" >&2; skipped=1; continue
    fi
    taken[$lower]=1
    echo "$b -> $lower"
    (( dry )) || mv -n -- "$dir/$b" "$dir/$lower"
done
exit $skipped
