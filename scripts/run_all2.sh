#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
run_one() {
  set -- $1
  kind=$1; model=$2; bench=$3; n=$4
  if [ "$kind" = "scale" ]; then
    python3 scripts/phase227_dwa_scale.py "$model" "$bench" "$n"
  else
    python3 scripts/phase225_newbench.py "$model" "$bench" "$n"
  fi
}
export -f run_one
xargs -P 2 -L 1 -I{} bash -c 'run_one "{}"' < /tmp/opencode/joblist3.txt
echo "ALL2 DONE $(date +%H:%M)" > logs/all2_done.flag
