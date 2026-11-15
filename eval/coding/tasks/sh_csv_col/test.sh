source ./assert.sh
check names "alice
bob \"b\" smith
carol
dave
eve
x" "$(bash solution.sh name fixture.csv)"
check by_index_quoted_header "10
2.50

-3
abc
.5" "$(bash solution.sh '#3' fixture.csv)"
check sum "count=4 sum=10 min=-3 max=10" "$(bash solution.sh -s 'amount, EUR' fixture.csv)"
check newline_value 'plain
multi\nline
a,b


z' "$(bash solution.sh note < fixture.csv)"
check none_numeric "count=0" "$(bash solution.sh -s name fixture.csv)"
printf 'a;b\n1;"x;y"\n2.5;z\n' > semi.csv
check delim "x;y
z" "$(bash solution.sh -d ';' b semi.csv)"
check delim_sum "count=2 sum=3.5 min=1 max=2.5" "$(bash solution.sh -d ';' -s a semi.csv)"
check unknown 3 "$(bash solution.sh nope fixture.csv >/dev/null 2>&1; echo $?)"
printf 'a\n"open\n' > bad.csv
check unterminated_rc 4 "$(bash solution.sh a bad.csv >/dev/null 2>&1; echo $?)"
check unterminated_out "" "$(bash solution.sh a bad.csv 2>/dev/null)"
finish
