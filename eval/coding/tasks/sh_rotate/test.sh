source ./assert.sh
D="my backups"
setup() {
    rm -rf "$D"; mkdir "$D"
    for ts in 20260105-010000 20260120-010000 20260203-010000 20260228-235959 20260301-000000 \
              20260415-120000 20260416-120000 20260417-120000 20260418-120000 20260419-120000; do
        touch "$D/db.main+1-$ts.tar.gz"
    done
    touch "$D/db.main+1-20251231-000000.tar.gz.part" "$D/dbXmain+1-20250101-000000.tar.gz" \
          "$D/other-20250101-000000.tar.gz" "$D/db.main+1-2026011-010000.tar.gz"
    mkdir "$D/db.main+1-20240101-000000.tar.gz"
    ln -s /etc/hostname "$D/db.main+1-20240102-000000.tar.gz"
    # mtimes deliberately inverted relative to names
    touch -d '2020-01-01' "$D/db.main+1-20260419-120000.tar.gz"
}
listing() { (cd "$D" && ls -1A | LC_ALL=C sort); }

setup; before=$(listing)
out=$(bash solution.sh -n -k 3 "$D" 'db.main+1')
check dry_run_out "delete db.main+1-20260105-010000.tar.gz
delete db.main+1-20260203-010000.tar.gz
delete db.main+1-20260415-120000.tar.gz
delete db.main+1-20260416-120000.tar.gz" "$out"
check dry_run_untouched "$before" "$(listing)"

bash solution.sh -k 3 "$D" 'db.main+1' >/dev/null
check deleted "$(LC_ALL=C sort <<<"db.main+1-2026011-010000.tar.gz
db.main+1-20240101-000000.tar.gz
db.main+1-20240102-000000.tar.gz
db.main+1-20251231-000000.tar.gz.part
db.main+1-20260120-010000.tar.gz
db.main+1-20260228-235959.tar.gz
db.main+1-20260301-000000.tar.gz
db.main+1-20260417-120000.tar.gz
db.main+1-20260418-120000.tar.gz
db.main+1-20260419-120000.tar.gz
dbXmain+1-20250101-000000.tar.gz
other-20250101-000000.tar.gz")" "$(listing)"
check symlink_target_intact 1 "$([[ -e /etc/hostname ]] && echo 1)"

setup
check default_keep "delete db.main+1-20260105-010000.tar.gz
delete db.main+1-20260203-010000.tar.gz" "$(bash solution.sh -n "$D" 'db.main+1')"
check keep_zero "delete db.main+1-20260105-010000.tar.gz
delete db.main+1-20260203-010000.tar.gz
delete db.main+1-20260415-120000.tar.gz
delete db.main+1-20260416-120000.tar.gz
delete db.main+1-20260417-120000.tar.gz
delete db.main+1-20260418-120000.tar.gz" "$(bash solution.sh -n -k 0 "$D" 'db.main+1')"
check nothing "" "$(bash solution.sh -k 3 "$D" nomatch)"
check missing_dir 2 "$(bash solution.sh -k 3 nodir x >/dev/null 2>&1; echo $?)"
check bad_keep 2 "$(bash solution.sh -k -1 "$D" x >/dev/null 2>&1; echo $?)"
check missing_args 2 "$(bash solution.sh "$D" >/dev/null 2>&1; echo $?)"
finish
