#!/usr/bin/env bash
# Fetch a file from HF without the broken systemd-resolved stub (see commit 16897fb).
# Usage: hfcurl.sh <repo> <path> [--dataset]
set -u
REPO="$1"; FILE="$2"; KIND="${3:-}"
ips(){ nslookup -type=A "$1" 1.1.1.1 2>/dev/null | awk '/^Address: /{print $2}' | grep -E '^[0-9.]+$'; }
RES=""
for h in huggingface.co us.aws.cdn.hf.co eu.aws.cdn.hf.co cdn-lfs.hf.co cdn-lfs-us-1.hf.co; do
  for ip in $(ips "$h"); do RES="$RES --resolve $h:443:$ip"; done
done
BASE="https://huggingface.co"
[ "$KIND" = "--dataset" ] && URL="$BASE/datasets/$REPO/resolve/main/$FILE" || URL="$BASE/$REPO/resolve/main/$FILE"
curl -4 -sL $RES --connect-timeout 20 --max-time 120 "$URL"
