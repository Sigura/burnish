PASSED=0
FAILED=0

check() {
    local name="$1" expected="$2" actual="$3"
    if [[ "$expected" == "$actual" ]]; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
        printf 'FAIL %s\n  expected: %q\n  actual:   %q\n' "$name" "$expected" "$actual"
    fi
}

finish() {
    echo "$PASSED/$((PASSED + FAILED)) passed"
    [[ $FAILED -eq 0 ]]
}
