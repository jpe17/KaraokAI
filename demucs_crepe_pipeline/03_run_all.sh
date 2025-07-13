#!/usr/bin/env bash
set -euo pipefail

### CONFIGURATION ###
IN_ROOT="02_data_augment"
OUT_ROOT="03_data_preprocessing"
# FFmpeg filter chain: resample→mono→80 Hz high-pass→–20 LUFS loudness
FILTER_CHAIN="aresample=44100, aformat=channel_layouts=mono, highpass=f=80, loudnorm=I=-20:TP=-1.5:LRA=7"

### CITATION NOTICE FOR GNU PARALLEL ###
# Suppress the first-run citation notice (one-time):
parallel --citation

### DETECT CPU CORES ###
if command -v nproc &>/dev/null; then
  NUM_CORES=$(nproc)
elif [[ "$(uname)" == "Darwin" ]]; then
  NUM_CORES=$(sysctl -n hw.logicalcpu)
else
  NUM_CORES=1
fi

### PREPARE DIRECTORIES ###
mkdir -p "${OUT_ROOT}/voice" "${OUT_ROOT}/instrumental"

### STEP 1: PARALLEL FFmpeg PREPROCESSING ###
FF_SCRIPT=$(mktemp)
cat > "${FF_SCRIPT}" << 'EOF'
#!/usr/bin/env bash
infile="$1"
stem=$(basename "$(dirname "$infile")")  # "voice" or "instrumental"
outfile="${OUT_ROOT}/${stem}/$(basename "$infile")"
ffmpeg -hide_banner -y -i "$infile" -af "${FILTER_CHAIN}" -ar 44100 "$outfile"
EOF
chmod +x "${FF_SCRIPT}"
export IN_ROOT OUT_ROOT FILTER_CHAIN FF_SCRIPT

find "${IN_ROOT}" -type f -name '*.wav' \
  | parallel --jobs "${NUM_CORES}" "${FF_SCRIPT}" {}

### STEP 2: GPU‐ACCELERATED DE‐REVERBERATION FOR VOICE STEMS ###
DEVERB_SCRIPT=$(mktemp)
cat > "${DEVERB_SCRIPT}" << 'EOF'
#!/usr/bin/env python3
import os, glob, torch, torchaudio
from openunmix import umxse

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
separator = umxse(pretrained=True, device=device)

in_dir = os.path.join("${OUT_ROOT}", "voice")
for path in glob.glob(os.path.join(in_dir, "*.wav")):
    wav, sr = torchaudio.load(path, normalize=True)
    wav = wav.mean(0, keepdim=True).to(device)
    with torch.no_grad():
        enhanced = separator(wav)
    torchaudio.save(path, enhanced.cpu(), sr)
    echo = f"De-reverbed: {os.path.basename(path)}"
    print(echo)
EOF
chmod +x "${DEVERB_SCRIPT}"
"${DEVERB_SCRIPT}"

### CLEANUP ###
rm "${FF_SCRIPT}" "${DEVERB_SCRIPT}"

echo "✅ All preprocessing complete."
