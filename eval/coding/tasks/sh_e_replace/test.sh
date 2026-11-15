source ./assert.sh
printf 'path a/b.c and a/b.c\nprice $5 & [x]\n' > one.txt
printf 'nothing here\n' > two.txt
printf 'no newline a/b.c' > three.txt
chmod 640 one.txt
out=$(bash solution.sh -b 'a/b.c' 'X&\1' one.txt two.txt three.txt); rc=$?
check rc 0 "$rc"
check out "one.txt: 2
three.txt: 1" "$out"
check one "path X&\\1 and X&\\1
price \$5 & [x]" "$(cat one.txt)"

check three_bytes "15" "$(wc -c < three.txt | tr -d ' ')"
check backup "path a/b.c and a/b.c" "$(head -1 one.txt.bak)"
check no_backup_unchanged 0 "$(ls two.txt.bak 2>/dev/null | wc -l)"
check perms 640 "$(stat -c %a one.txt)"
check special_old "1" "$(bash solution.sh '$5 & [x]' 'ok' one.txt | sed 's/.*: //')"
check special_res "price ok" "$(sed -n 2p one.txt)"
printf 'aaa\n' > dry.txt
check dry_out "dry.txt: 3" "$(bash solution.sh -n a b dry.txt)"
check dry_unchanged "aaa" "$(cat dry.txt)"
check none_rc 1 "$(bash solution.sh zzz y two.txt >/dev/null 2>&1; echo $?)"
check missing_rc 2 "$(bash solution.sh a b dry.txt nofile >/dev/null 2>&1; echo $?)"
check missing_untouched "aaa" "$(cat dry.txt)"
check empty_old 2 "$(bash solution.sh '' b dry.txt >/dev/null 2>&1; echo $?)"
finish
