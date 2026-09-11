# usage: fetch_small.sh <model> ; reads URLs from stdin lines "name url"
M=$1; mkdir -p results/artifacts/$M/run
while read name url; do curl -s -o results/artifacts/$M/run/$name "$url" && echo "$name $(stat -c %s results/artifacts/$M/run/$name)"; done
