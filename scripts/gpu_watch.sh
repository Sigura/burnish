#!/usr/bin/env bash
# Logs per-GPU clocks, power, temperature and throttle reasons every INTERVAL seconds to a CSV, and appends one
# line to an alerts file when a GPU throttles abnormally (hardware/thermal slowdown, high temperature, or a low
# SM clock while busy). Alerts repeat at most once per GPU and reason every ALERT_EVERY seconds.
# Usage: scripts/gpu_watch.sh [OUT_DIR]   (default run-logs/gpu-watch)
set -euo pipefail

OUT="${1:-run-logs/gpu-watch}"
INTERVAL="${INTERVAL:-30}"
ALERT_EVERY="${ALERT_EVERY:-600}"
TEMP_WARN="${TEMP_WARN:-83}"
# busy (util >= 50%) but SM clock below this fraction of its max
CLOCK_FRAC="${CLOCK_FRAC:-0.5}"

echo "== checking prerequisites"
command -v nvidia-smi >/dev/null || { echo "missing nvidia-smi" >&2; exit 1; }
mkdir -p "$OUT"
CSV="$OUT/gpu.csv"
ALERTS="$OUT/alerts.log"
FIELDS=index,name,utilization.gpu,clocks.sm,clocks.max.sm,clocks.mem,power.draw,power.limit,temperature.gpu,temperature.memory,fan.speed,clocks_event_reasons.active
[[ -s $CSV ]] || echo "time,$FIELDS" > "$CSV"
touch "$ALERTS"

echo "== logging to $CSV every ${INTERVAL}s, alerts to $ALERTS"
declare -A last=() lowrun=()
while true; do
    now=$(date +%s)
    ts=$(date +%FT%T)
    while IFS=, read -r idx name util sm smmax mem pwr plim temp tmem fan reasons; do
        echo "$ts,$idx,$name,$util,$sm,$smmax,$mem,$pwr,$plim,$temp,$tmem,$fan,$reasons" >> "$CSV"
        r=$(( reasons ))
        problems=()
        (( r & 0x08 )) && problems+=("hw-slowdown")
        (( r & 0x20 )) && problems+=("sw-thermal")
        (( r & 0x40 )) && problems+=("hw-thermal")
        (( r & 0x80 )) && problems+=("hw-power-brake")
        [[ $temp =~ ^[0-9]+$ ]] && (( temp >= TEMP_WARN )) && problems+=("temp-${temp}C")
        # a single low sample is common with bursty partial-offload loads, so require two in a row
        if [[ $util =~ ^[0-9]+$ && $sm =~ ^[0-9]+$ && $smmax =~ ^[0-9]+$ ]] && (( util >= 50 )) \
            && awk -v a="$sm" -v b="$smmax" -v f="$CLOCK_FRAC" 'BEGIN { exit !(a < b * f) }'; then
            (( ${lowrun[$idx]:-0} >= 1 )) && problems+=("low-clock-${sm}of${smmax}MHz")
            lowrun[$idx]=$(( ${lowrun[$idx]:-0} + 1 ))
        else
            lowrun[$idx]=0
        fi
        for p in "${problems[@]}"; do
            key="$idx:${p%%-[0-9]*}"
            if (( now - ${last[$key]:-0} >= ALERT_EVERY )); then
                last[$key]=$now
                echo "$ts GPU $idx ($name) $p util=${util}% sm=${sm}/${smmax}MHz power=${pwr}/${plim}W temp=${temp}C reasons=$reasons" >> "$ALERTS"
            fi
        done
    done < <(nvidia-smi --query-gpu=$FIELDS --format=csv,noheader,nounits | sed 's/, /,/g')
    sleep "$INTERVAL"
done
