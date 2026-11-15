#!/usr/bin/env bash
set -euo pipefail
file=${1:?usage: $0 FILE}
declare -A val=() last=()
order=()
n=0
fail() { echo "$file:$n: invalid line" >&2; exit 1; }
rest_ok() { [[ $1 =~ ^[[:blank:]]*(#.*)?$ ]]; }
while IFS= read -r line || [[ -n $line ]]; do
    n=$((n + 1))
    [[ $line =~ ^[[:blank:]]*(#.*)?$ ]] && continue
    line=${line#export }
    [[ $line =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || fail
    key=${BASH_REMATCH[1]} v=${BASH_REMATCH[2]}
    if [[ $v == \'* ]]; then
        [[ $v =~ ^\'([^\']*)\'(.*)$ ]] || fail
        v=${BASH_REMATCH[1]}
        rest_ok "${BASH_REMATCH[2]}" || fail
    elif [[ $v == \"* ]]; then
        body=${v:1} out= i=0 closed=0
        while (( i < ${#body} )); do
            c=${body:i:1}
            if [[ $c == '\' && $((i + 1)) -lt ${#body} ]]; then
                nx=${body:i+1:1}
                case $nx in
                    n) out+=$'\n' ;;
                    '"') out+='"' ;;
                    '\') out+='\' ;;
                    *) out+="\\$nx" ;;
                esac
                i=$((i + 2))
            elif [[ $c == '"' ]]; then
                closed=1; break
            else
                out+=$c; i=$((i + 1))
            fi
        done
        (( closed )) || fail
        rest_ok "${body:i+1}" || fail
        v=$out
    else
        v=${v%%[[:blank:]]#*}
        v=${v%"${v##*[![:blank:]]}"}
    fi
    val[$key]=$v
    last[$key]=$n
    order+=("$key")
done < "$file"
for i in "${!order[@]}"; do
    k=${order[i]}
    printf '%s\n' "$k"
done | awk '{ pos[$0] = NR; keys[NR] = $0 } END { for (i = 1; i <= NR; i++) if (pos[keys[i]] == i) print keys[i] }' \
| while IFS= read -r k; do
    v=${val[$k]}
    printf "export %s='%s'\n" "$k" "${v//\'/\'\\\'\'}"
done
