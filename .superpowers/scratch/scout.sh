#!/bin/bash
# For each symbol: HEAD the 2023-06 aggTrades monthly zip; print "symbol size_bytes" if it exists
sym="$1"
sz=$(curl -sI "https://data.binance.vision/data/futures/um/monthly/aggTrades/${sym}/${sym}-aggTrades-2023-06.zip" | grep -i '^content-length' | tr -d '\r' | awk '{print $2}')
code=$(curl -s -o /dev/null -w '%{http_code}' -I "https://data.binance.vision/data/futures/um/monthly/aggTrades/${sym}/${sym}-aggTrades-2023-06.zip")
if [ "$code" = "200" ] && [ -n "$sz" ]; then echo "$sym $sz"; fi
