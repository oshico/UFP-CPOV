#!/usr/bin/env bash
# Download UCF101 (~6.5 GB) and extract only the 10 project classes into $1/UCF-101 (default: data).
set -euo pipefail
DEST=${1:-data}
CLASSES="ApplyLipstick BrushingTeeth PlayingGuitar PlayingViolin Typing WritingOnBoard PushUps JumpingJack CuttingInKitchen ApplyEyeMakeup"
URL=https://www.crcv.ucf.edu/data/UCF101/UCF101.rar

mkdir -p "$DEST"
[ -f "$DEST/UCF101.rar" ] || wget -q --show-progress --no-check-certificate -O "$DEST/UCF101.rar" "$URL"
command -v unrar >/dev/null || apt-get -qq install -y unrar
unrar x -o+ -idq "$DEST/UCF101.rar" $(for c in $CLASSES; do printf "UCF-101/%s/* " "$c"; done) "$DEST/"
for c in $CLASSES; do printf "%-18s %s videos\n" "$c" "$(ls "$DEST/UCF-101/$c" | wc -l)"; done
