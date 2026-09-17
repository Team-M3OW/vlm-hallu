#!/bin/bash
# gpuwait.sh <MiB needed> <timeout s> -- block until that much VRAM is free (external jobs included)
need=$1; deadline=$(( $(date +%s) + ${2:-7200} ))
while :; do
  free=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) ))
  [ "$free" -ge "$need" ] && { echo "gpuwait: ${free}MiB free >= ${need}MiB, proceeding"; exit 0; }
  [ "$(date +%s)" -ge "$deadline" ] && { echo "gpuwait: TIMEOUT, only ${free}MiB free"; exit 1; }
  sleep 60
done
