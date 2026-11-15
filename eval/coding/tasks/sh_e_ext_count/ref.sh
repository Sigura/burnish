#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 && -d $1 && ! -L $1 ]] || { echo "usage: $0 DIR" >&2; exit 1; }
find "$1" \( -name .git -o -name node_modules \) -type d -prune -o -type f -print0 \
| while IFS= read -r -d '' f; do
    b=${f##*/}
    s=${b#.}
    if [[ $s == *.* ]]; then e=${s##*.}; printf '%s\n' "${e,,}"; else echo '(none)'; fi
done | LC_ALL=C sort | uniq -c | awk '{print $2, $1}' | LC_ALL=C sort -k2,2nr -k1,1
