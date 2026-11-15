#!/usr/bin/env bash
set -euo pipefail
for f in "$@"; do
    [[ $f == - || -r $f ]] || { echo "cannot read $f" >&2; exit 1; }
done
export LC_ALL=C
for f in "$@"; do
    case $f in
        -) cat ;;
        *.gz) zcat -- "$f" ;;
        *) cat -- "$f" ;;
    esac
done | awk '
    match($0, /^[^ ]+ [^ ]+ [^ ]+ \[[^]]+\] "[A-Z]+ [^ "]+ [^ "]+" [0-9][0-9][0-9] ([0-9]+|-) "[^"]*" "[^"]*"$/) {
        split($0, q, "\"")
        split(q[2], req, " ")
        path = req[2]; sub(/\?.*/, "", path)
        n = split(q[3], ss, " ")
        code = ss[1]; size = (ss[2] == "-") ? 0 : ss[2]
        ok++; st[code]++; ipb[$1] += size; pc[path]++
        next
    }
    { bad++ }
    END {
        print "requests: " ok + 0
        print "malformed: " bad + 0
        for (c in st) print "S", c, st[c]
        for (i in ipb) print "I", i, ipb[i]
        for (p in pc) print "P", p, pc[p]
    }' > "${TMPDIR:-/tmp}/al.$$"
head -2 "${TMPDIR:-/tmp}/al.$$"
echo "status:"
awk '$1 == "S"' "${TMPDIR:-/tmp}/al.$$" | sort -k2,2n | awk '{print "  " $2, $3}'
echo "top_ips:"
awk '$1 == "I"' "${TMPDIR:-/tmp}/al.$$" | sort -k3,3nr -k2,2 | head -3 | awk '{print "  " $2, $3}'
echo "top_paths:"
awk '$1 == "P"' "${TMPDIR:-/tmp}/al.$$" | sort -k3,3nr -k2,2 | head -3 | awk '{print "  " $2, $3}'
rm -f "${TMPDIR:-/tmp}/al.$$"
