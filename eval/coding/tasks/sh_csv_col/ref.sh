#!/usr/bin/env bash
set -euo pipefail
delim=, stats=0
while getopts ":d:s" o; do
    case $o in
        d) delim=$OPTARG ;;
        s) stats=1 ;;
        *) echo "usage: $0 [-d DELIM] [-s] COLUMN [FILE]" >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))
col=${1:?usage}
file=${2:--}
out=$(cat -- "$file" | awk -v D="$delim" -v COL="$col" -v STATS="$stats" '
BEGIN { RS = "\001"; }
{
    s = $0; n = length(s); i = 1; row = 0; nf = 0; f = ""; inq = 0; started = 0
    while (i <= n) {
        c = substr(s, i, 1)
        if (inq) {
            if (c == "\"") {
                if (substr(s, i + 1, 1) == "\"") { f = f "\""; i += 2; continue }
                inq = 0; i++; continue
            }
            if (c == "\r" && substr(s, i + 1, 1) == "\n") { f = f "\\n"; i += 2; continue }
            if (c == "\n") { f = f "\\n"; i++; continue }
            f = f c; i++; continue
        }
        if (c == "\"") { inq = 1; started = 1; i++; continue }
        if (c == D) { fld[++nf] = f; f = ""; started = 1; i++; continue }
        if (c == "\r" && substr(s, i + 1, 1) == "\n") { i++; continue }
        if (c == "\n") { fld[++nf] = f; endrow(); f = ""; nf = 0; started = 0; i++; continue }
        f = f c; started = 1; i++
    }
    if (inq) exit 4
    if (started || nf > 0) { fld[++nf] = f; endrow() }
    finish()
}
function endrow(   k) {
    row++
    if (row == 1) {
        idx = 0
        if (COL ~ /^#[0-9]+$/) idx = substr(COL, 2) + 0
        else for (k = 1; k <= nf; k++) if (fld[k] == COL) { idx = k; break }
        if (idx < 1 || idx > nf) { bad = 1; exit 3 }
        return
    }
    v = (idx <= nf) ? fld[idx] : ""
    for (k = 1; k <= nf; k++) delete fld[k]
    if (!STATS) { vals[++nv] = v; return }
    if (v ~ /^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)$/) {
        x = v + 0; cnt++; sum += x
        if (cnt == 1 || x < mn) mn = x
        if (cnt == 1 || x > mx) mx = x
    }
}
function finish(   k) {
    if (!STATS) { for (k = 1; k <= nv; k++) print vals[k]; return }
    if (cnt == 0) print "count=0"
    else printf "count=%d sum=%g min=%g max=%g\n", cnt, sum, mn, mx
}
') || rc=$?
case ${rc:-0} in
    0) printf '%s\n' "$out" ;;
    3) echo "unknown column: $col" >&2; exit 3 ;;
    4) echo "unterminated quote" >&2; exit 4 ;;
    *) exit "$rc" ;;
esac
