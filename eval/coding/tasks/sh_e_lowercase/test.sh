source ./assert.sh
setup() {
    rm -rf d; mkdir d
    touch d/README.md "d/My File.TXT" d/already.txt d/Dup.txt d/DUP.txt d/Taken.log d/taken.log d/ÄBC.Txt
    mkdir d/SubDir; ln -s README.md d/Link.MD
}
setup
out=$(bash solution.sh -n d 2>err); rc=$?
check dry_out "DUP.txt -> dup.txt
My File.TXT -> my file.txt
README.md -> readme.md
ÄBC.Txt -> Äbc.txt" "$out"
check dry_rc 1 "$rc"
check dry_err "skip: Dup.txt (exists)
skip: Taken.log (exists)" "$(LC_ALL=C sort err)"
check dry_nothing_renamed 1 "$([[ -e d/README.md && ! -e d/readme.md ]] && echo 1)"
bash solution.sh d >/dev/null 2>&1
check listing "Dup.txt
Link.MD
SubDir
Taken.log
already.txt
dup.txt
my file.txt
readme.md
taken.log
Äbc.txt" "$(cd d && ls -1A | LC_ALL=C sort)"
rm -rf e; mkdir e; touch e/A
check clean_rc 0 "$(bash solution.sh e >/dev/null; echo $?)"
check missing 2 "$(bash solution.sh nodir >/dev/null 2>&1; echo $?)"
finish
