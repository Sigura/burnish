source ./assert.sh
cat > flaky.sh <<'X'
#!/usr/bin/env bash
n=$(cat count 2>/dev/null || echo 0); n=$((n + 1)); echo $n > count
echo "run $n"
[[ $n -ge ${SUCCEED_AT:-99} ]] && exit 0
exit ${CODE:-3}
X
chmod +x flaky.sh
reset() { rm -f count; }

reset; out=$(SUCCEED_AT=3 bash solution.sh -d 0.01 -- ./flaky.sh 2>err); rc=$?
check succeed_rc 0 "$rc"
check succeed_out "run 1
run 2
run 3" "$out"
check succeed_err "retry: attempt 1/3 failed with exit code 3, sleeping 0.01 s
retry: attempt 2/3 failed with exit code 3, sleeping 0.02 s" "$(cat err)"

reset; bash solution.sh -n 4 -d 0.01 -b 1.5 -- ./flaky.sh >/dev/null 2>err; rc=$?
check fail_rc 3 "$rc"
check fail_err "retry: attempt 1/4 failed with exit code 3, sleeping 0.01 s
retry: attempt 2/4 failed with exit code 3, sleeping 0.015 s
retry: attempt 3/4 failed with exit code 3, sleeping 0.0225 s" "$(cat err)"
check fail_runs 4 "$(cat count)"

reset; CODE=5 bash solution.sh -r 3,4 -d 0.01 -- ./flaky.sh >/dev/null 2>err; rc=$?
check nonretryable_rc 5 "$rc"
check nonretryable_runs 1 "$(cat count)"
check nonretryable_err "" "$(cat err)"

reset; CODE=4 bash solution.sh -n 2 -r 3,4 -d 0.01 -- ./flaky.sh >/dev/null 2>/dev/null
check retryable_listed 2 "$(cat count)"

check args_preserved "[a b] [] [*] [\$HOME]" "$(bash solution.sh -n 1 -- printf '[%s] ' 'a b' '' '*' '$HOME' | sed 's/ $//')"
check stdin_first_only "x||" "$(cat > rd.sh <<'X'
read -r line; printf '%s|' "$line"; exit 1
X
printf "x\ny\n" | bash solution.sh -n 2 -d 0.01 -- bash rd.sh 2>/dev/null)"

start=$(date +%s%N); reset; bash solution.sh -n 3 -d 0.2 -b 1 -- ./flaky.sh >/dev/null 2>&1; end=$(date +%s%N)
ms=$(( (end - start) / 1000000 ))
check sleeps_happen 1 "$(( ms >= 380 && ms < 2000 ))"

check no_command 64 "$(bash solution.sh -n 2 >/dev/null 2>&1; echo $?)"
check no_dashdash 64 "$(bash solution.sh -n 2 true >/dev/null 2>&1; echo $?)"
check bad_n 64 "$(bash solution.sh -n x -- true >/dev/null 2>&1; echo $?)"
check bad_d 64 "$(bash solution.sh -d abc -- true >/dev/null 2>&1; echo $?)"
check default_success 0 "$(bash solution.sh -- true >/dev/null 2>&1; echo $?)"
finish
