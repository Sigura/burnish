#!/usr/bin/env bash
set -uo pipefail
usage() { echo "usage: $0 [-n N] [-d DELAY] [-b FACTOR] [-r CODES] -- COMMAND [ARGS...]" >&2; exit 64; }
n=3 d=1 b=2 codes=
num='^[0-9]+([.][0-9]+)?$|^[.][0-9]+$'
while (( $# )); do
    case $1 in
        -n) [[ ${2-} =~ ^[1-9][0-9]*$ ]] || usage; n=$2; shift 2 ;;
        -d) [[ ${2-} =~ $num ]] || usage; d=$2; shift 2 ;;
        -b) [[ ${2-} =~ $num ]] || usage; b=$2; shift 2 ;;
        -r) [[ ${2-} =~ ^[0-9]+(,[0-9]+)*$ ]] || usage; codes=",$2,"; shift 2 ;;
        --) shift; break ;;
        *) usage ;;
    esac
done
(( $# )) || usage
k=1
while :; do
    if (( k == 1 )); then "$@"; else "$@" </dev/null; fi
    rc=$?
    (( rc == 0 )) && exit 0
    (( k >= n )) && exit "$rc"
    [[ -z $codes || $codes == *",$rc,"* ]] || exit "$rc"
    s=$(awk -v d="$d" 'BEGIN { printf "%g", d }')
    echo "retry: attempt $k/$n failed with exit code $rc, sleeping $s s" >&2
    sleep "$d"
    d=$(awk -v d="$d" -v b="$b" 'BEGIN { printf "%.10g", d * b }')
    k=$((k + 1))
done
