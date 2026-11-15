source ./assert.sh
gzip -c fixture_b.log > b.log.gz
check two_files "requests: 7
malformed: 2
status:
  200 4
  201 1
  304 1
  404 1
top_ips:
  203.0.113.9 5120
  10.0.0.1 1000
  198.51.100.2 300
top_paths:
  /api/v1/items 3
  /health 2
  /api/v1/items/7 1" "$(bash solution.sh fixture_a.log b.log.gz)"
check stdin_and_ties "requests: 3
malformed: 0
status:
  200 2
  404 1
top_ips:
  10.0.0.3 122
  10.0.0.2 2
top_paths:
  /health 2
  /api/v1/items/7 1" "$(bash solution.sh - < fixture_b.log)"
check only_malformed "requests: 0
malformed: 1
status:
top_ips:
top_paths:" "$(echo 'nope' | bash solution.sh -)"
check missing_rc 1 "$(bash solution.sh nope.log >/dev/null 2>&1; echo $?)"
check missing_out "" "$(bash solution.sh fixture_a.log nope.log 2>/dev/null)"
finish
