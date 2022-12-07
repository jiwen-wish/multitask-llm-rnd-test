#!/bin/bash -e
rm -rf memory.log
echo "      date     time $(free -m | grep total | sed -E 's/^    (.*)/\1/g')" >> memory.log
while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') $(free -hm | grep Mem: | sed 's/Mem://g')" >> memory.log
    sleep 1m
done