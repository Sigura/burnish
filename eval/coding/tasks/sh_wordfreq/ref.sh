#!/usr/bin/env bash
set -euo pipefail
usage() { echo "usage: $0 [-n N] [-i]" >&2; exit 2; }
n=10 ci=0
while getopts ":n:i" opt; do
    case $opt in
        n) n=$OPTARG ;;
        i) ci=1 ;;
        *) usage ;;
    esac
done
[[ $n =~ ^[0-9]+$ ]] && (( n > 0 )) || usage
export LC_ALL=C
tr -c "A-Za-z0-9'" '\n' | sed "s/^'*//; s/'*\$//" | grep -v '^$' \
    | { if (( ci )); then tr 'A-Z' 'a-z'; else cat; fi; } \
    | sort | uniq -c | awk '{print $1, $2}' | sort -k1,1nr -k2,2 | head -n "$n" || true
