#!/usr/bin/env bash
set -euo pipefail

IN_ROOT="02_data_augment"
OUT_ROOT="03_data_preprocessing"
FILTER_CHAIN="aresample=44100, aformat=channel_layouts=mono, highpass=f=80:order=2, loudnorm=I=-20:TP=-1.5:LRA=7"

mkdir -p "${OUT_ROOT}/voice" "${OUT_ROOT}/instrumental"
export IN_ROOT OUT_ROOT FILTER_CHAIN

find "${IN_ROOT}" -type f -name '*.wav' \
  | parallel --jobs "$(nproc)" \
    'stem=$(basename "$(dirname "{}")"); \
     ffmpeg -hide_banner -y -i "{}" -af "$FILTER_CHAIN" -ar 44100 "${OUT_ROOT}/${stem}/$(basename "{}")"'
