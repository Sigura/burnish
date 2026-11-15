#!/usr/bin/env bash
set -euo pipefail
die() { echo "$*" >&2; exit 2; }
dry=0 keep=7
while getopts ":nk:" o; do
    case $o in
        n) dry=1 ;;
        k) keep=$OPTARG ;;
        *) die "usage: $0 [-n] [-k KEEP] DIR PREFIX" ;;
    esac
done
shift $((OPTIND - 1))
(( $# == 2 )) || die "usage: $0 [-n] [-k KEEP] DIR PREFIX"
[[ $keep =~ ^[0-9]+$ ]] || die "bad KEEP"
dir=$1 prefix=$2
[[ -d $dir ]] || die "no such dir: $dir"
names=()
for f in "$dir"/*; do
    b=${f##*/}
    [[ -f $f && ! -L $f ]] || continue
    [[ ${b:0:${#prefix}} == "$prefix" ]] || continue
    rest=${b:${#prefix}}
    [[ $rest =~ ^-[0-9]{8}-[0-9]{6}\.tar\.gz$ ]] || continue
    names+=("${rest:1:15} $b")
done
(( ${#names[@]} )) || exit 0
mapfile -t sorted < <(printf '%s\n' "${names[@]}" | LC_ALL=C sort -r)
declare -A keepset=() months=()
i=0
for line in "${sorted[@]}"; do
    ts=${line%% *} b=${line#* }
    (( i < keep )) && keepset[$b]=1
    m=${ts:0:6}
    if [[ -z ${months[$m]-} ]] && (( ${#months[@]} < 4 )); then
        months[$m]=1; keepset[$b]=1
    fi
    i=$((i + 1))
done
for (( j = ${#sorted[@]} - 1; j >= 0; j-- )); do
    b=${sorted[j]#* }
    [[ -n ${keepset[$b]-} ]] && continue
    echo "delete $b"
    (( dry )) || rm -- "$dir/$b"
done
