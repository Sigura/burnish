source ./assert.sh
out=$(bash solution.sh fixture.env); rc=$?
check rc 0 "$rc"
check lines "export A
export C
export D
export E
export F
export G
export B
export H
export I" "$(sed -n 's/^\(export [A-Z]*\)=.*/\1/p' <<<"$out")"
eval "$out"
check A "plain value" "$A"
check B "second" "$B"
check C "a  b " "$C"
check D 'single $HOME `x` "q"' "$D"
check E $'line1\nline2 "q" back\\slash \\t $HOME' "$E"
check F "" "$F"
check G "" "$G"
check H "x'y" "$H"
check I "	spaced" "$I"
check line_count 1 "$(grep -c '^export B=' <<<"$out")"
check quoting_single 1 "$(grep -c "^export H='x'\\\\''y'$" <<<"$out")"

printf 'OK=1\nexport\nX=2\n' > bad1.env
out=$(bash solution.sh bad1.env 2>err); rc=$?
check bad_rc 1 "$rc"
check bad_out "" "$out"
check bad_err "bad1.env:2: invalid line" "$(cat err)"
for bad in 'K = v' '1K=v' "K='open" 'K="open' "K='a' junk" 'K="a"b' 'K'; do
    printf 'A=1\n%s\n' "$bad" > bad.env
    check "bad[$bad]" 1 "$(bash solution.sh bad.env >/dev/null 2>&1; echo $?)"
done
printf 'K="a" # ok\nL=x # y # z\n' > ok.env
eval "$(bash solution.sh ok.env)"
check comment_after_quote "a" "$K"
check comment_unquoted "x" "$L"
finish
