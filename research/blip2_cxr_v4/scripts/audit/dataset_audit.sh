#!/usr/bin/env bash
set -u
timeout 60 find /storage/shared -maxdepth 2 -type d \( -iname '*mimic*' -o -iname '*minic*' -o -iname '*vindr*' -o -iname '*vindxr*' -o -iname '*indiana*' -o -iname '*iu*' -o -iname '*openi*' -o -iname '*chexpert*' -o -iname '*nih*' -o -iname '*chest*x*ray*14*' -o -iname '*padchest*' \) 2>/dev/null | head -200
for root in /storage/shared/MIMIC-CXR-JPG /storage/shared/MIMIC-CXR-Report /storage/shared/VinDr-CXR /storage/shared/MODELS /storage/shared/sontg/NIH-Chest-X-rays /storage/shared/sontg/CheXpert /storage/shared/sontg/PadChest; do
  echo "ROOT $root"
  timeout 20 ls "$root" 2>/dev/null | head -25
done
