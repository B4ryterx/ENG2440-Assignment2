#!/usr/bin/env bash
# Downloads MSD Task01_BrainTumour (~7.6 GB) from the MONAI mirror and extracts it here.
# Safe to re-run: resumes a partial download and retries automatically if the connection drops.
set -uo pipefail
URL="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
OUT="Task01_BrainTumour.tar"

if pgrep -f "curl.*$OUT" >/dev/null; then
  echo "Another download of $OUT is already running — not starting a second one."; exit 1
fi

EXPECTED=$(curl -sI "$URL" | awk 'tolower($1)=="content-length:"{print $2}' | tr -d '\r')
for attempt in $(seq 1 50); do
  curl -L -C - --retry 5 --retry-delay 5 -o "$OUT" "$URL" && break
  echo "Connection dropped (attempt $attempt) — resuming in 10 s..."; sleep 10
done

SIZE=$(stat -f%z "$OUT" 2>/dev/null || stat -c%s "$OUT")
if [ -n "$EXPECTED" ] && [ "$SIZE" != "$EXPECTED" ]; then
  echo "Incomplete: $SIZE / $EXPECTED bytes. Re-run this script to resume."; exit 1
fi

tar -xf "$OUT" && echo "Extracted." 
echo "Cases in imagesTr: $(ls Task01_BrainTumour/imagesTr | grep -c '^BRATS_.*\.nii\.gz$')  (expect 484)"
