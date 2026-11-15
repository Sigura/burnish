#!/usr/bin/env bash
set -euo pipefail
die() { echo "$*" >&2; exit 2; }
dry=0 bak=0
while getopts ":nb" o; do
    case $o in n) dry=1 ;; b) bak=1 ;; *) die "usage: $0 [-n] [-b] OLD NEW FILE..." ;; esac
done
shift $((OPTIND - 1))
(( $# >= 3 )) || die "usage: $0 [-n] [-b] OLD NEW FILE..."
old=$1 new=$2; shift 2
[[ -n $old && $old != *$'\n'* ]] || die "bad OLD"
for f in "$@"; do [[ -f $f ]] || die "no such file: $f"; done
changed=0
for f in "$@"; do
    content=$(cat -- "$f"; printf x); content=${content%x}
    rest=$content n=0
    while [[ $rest == *"$old"* ]]; do rest=${rest#*"$old"}; n=$((n + 1)); done
    (( n )) || continue
    changed=1
    echo "$f: $n"
    (( dry )) && continue
    (( bak )) && cp -p -- "$f" "$f.bak"
    printf '%s' "${content//"$old"/"$new"}" > "$f"
done
(( changed )) && exit 0 || exit 1
