source ./assert.sh
run() { bash solution.sh "$@"; }

check basic "3 the
2 cat
1 a
1 sat" "$(printf 'the cat sat\nthe cat, a the\n' | run -n 4)"
check default_n "$(seq 1 12 | awk '{print 13-$1}' | sort -n | sed 's/^/1 /' | LC_ALL=C sort -k2,2 | head -10)" "$(seq 1 12 | run)"
check case "2 Foo
1 foo" "$(printf 'Foo foo Foo' | run)"
check case_i "3 foo" "$(printf 'Foo foo FOO' | run -i)"
check apostrophes "2 don't
1 rock
1 rock'n'roll" "$(printf "'don't' don't ''' rock'n'roll 'rock'" | run)"
check tie_order "1 B
1 a
1 b" "$(printf 'b a B' | run)"
check fewer "1 x" "$(printf 'x' | run -n 5)"
check empty "" "$(printf '' | run)"
check empty_rc 0 "$(printf '' | run >/dev/null; echo $?)"
check digits "2 42
1 x2" "$(printf '42 x2 42' | run)"
check bad_n 2 "$(echo a | run -n 0 >/dev/null 2>&1; echo $?)"
check bad_n2 2 "$(echo a | run -n abc >/dev/null 2>&1; echo $?)"
check bad_opt 2 "$(echo a | run -z >/dev/null 2>&1; echo $?)"
check bad_stderr 1 "$(echo a | run -z 2>&1 >/dev/null | grep -c .)"
finish
