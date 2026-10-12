#!/usr/bin/env bash
# Robinson setup — Raspberry Pi 5, Raspberry Pi OS Lite (trixie, arm64).
# Installs system deps, the Python env (uv), all models, enables I2C and the
# systemd service. Safe to re-run.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT=$PWD
MODELS=${ROBINSON_MODELS_DIR:-$HOME/robinson-models}
export PATH="$HOME/.local/bin:$PATH"

step() { echo -e "\n\033[0;32m=== $* ===\033[0m"; }
fetch() {  # url dest — skip if present
    [ -s "$2" ] && return 0
    echo "→ $(basename "$2")"
    curl -fL --progress-bar -o "$2.part" "$1" && mv "$2.part" "$2"
}

step "System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq curl ca-certificates libportaudio2 libsndfile1 alsa-utils \
    i2c-tools python3-dev build-essential cmake swig liblgpio-dev

step "I2C (PCA9685 servo driver)"
if ! grep -q '^dtparam=i2c_arm=on' /boot/firmware/config.txt; then
    sudo raspi-config nonint do_i2c 0
    echo "I2C enabled — reboot required before the servos work"
fi
sudo usermod -aG i2c,audio,gpio "$USER"

step "uv + Python environment"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

step "Models → $MODELS"
mkdir -p "$MODELS/cache"
fetch https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm \
      "$MODELS/gemma-4-E2B-it.litertlm"
fetch https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad_v5.onnx \
      "$MODELS/silero_vad_v5.onnx"
if [ ! -d "$MODELS/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8" ]; then
    echo "→ parakeet-tdt-0.6b-v3-int8"
    curl -fL --progress-bar https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2 \
        | tar xj -C "$MODELS"
fi
V=https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten_emotional/medium
fetch "$V/de_DE-thorsten_emotional-medium.onnx" "$MODELS/de_DE-thorsten_emotional-medium.onnx"
fetch "$V/de_DE-thorsten_emotional-medium.onnx.json" "$MODELS/de_DE-thorsten_emotional-medium.onnx.json"

step "systemd service"
sed -e "s|@USER@|$USER|g" -e "s|@ROOT@|$ROOT|g" deploy/robinson.service \
    | sudo tee /etc/systemd/system/robinson.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable robinson.service

echo -e "\n\033[0;32m✓ Setup complete.\033[0m"
echo "  start:  sudo systemctl start robinson   (logs: journalctl -u robinson -f)"
echo "  debug:  uv run robinson --text           (type instead of talk)"
