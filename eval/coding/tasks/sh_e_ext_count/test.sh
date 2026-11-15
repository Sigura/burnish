source ./assert.sh
D="tree dir"
mkdir -p "$D/src/deep" "$D/.git/objects" "$D/web/node_modules/pkg" "$D/docs"
touch "$D/src/a.py" "$D/src/b.PY" "$D/src/deep/c.py" "$D/src/deep/x.tar.gz" "$D/docs/read me.md" \
      "$D/docs/notes.MD" "$D/Makefile" "$D/.bashrc" "$D/.git/objects/zz.py" "$D/web/node_modules/pkg/i.js" \
      "$D/web/app.js" "$D/docs/$(printf 'new\nline.txt')"
ln -s ../src/a.py "$D/docs/link.py"
mkdir "$D/dir.py"
check counts "py 3
(none) 2
md 2
gz 1
js 1
txt 1" "$(bash solution.sh "$D")"
check missing 1 "$(bash solution.sh nope >/dev/null 2>&1; echo $?)"
check file_arg 1 "$(bash solution.sh "$D/Makefile" >/dev/null 2>&1; echo $?)"
mkdir empty
check empty "" "$(bash solution.sh empty)"
finish
