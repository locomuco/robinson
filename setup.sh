#!/usr/bin/env bash
# Robinson setup script — Raspberry Pi 5 / Ubuntu aarch64
# Installs: uv · ollama · whisper.cpp · piper
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OLLAMA_MODEL="qwen2.5:1.5b"
WHISPER_MODEL="base.en"
PIPER_RELEASE="2023.11.14-2"
PIPER_VOICE_NAME="en_US-lessac-medium"
PIPER_HF_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

step() { echo -e "\n${GREEN}=== $* ===${NC}"; }
info() { echo -e "${YELLOW}→ $*${NC}"; }

# ── 1. System dependencies ────────────────────────────────────────────────────
step "System dependencies"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    cmake \
    build-essential \
    git \
    wget \
    curl \
    portaudio19-dev \
    libsndfile1-dev \
    python3-dev

# ── 2. uv ─────────────────────────────────────────────────────────────────────
step "uv (Python package manager)"
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Make uv available for the rest of this script
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
    info "uv installed — you may need to restart your shell or run: source ~/.bashrc"
else
    info "uv already installed: $(uv --version)"
fi

# ── 3. ollama ─────────────────────────────────────────────────────────────────
step "ollama"
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
else
    info "ollama already installed"
fi

# Start ollama in background if not already running
if ! pgrep -x ollama &>/dev/null; then
    info "Starting ollama server …"
    ollama serve &>/tmp/ollama.log &
    OLLAMA_PID=$!
    # Wait for it to be ready
    for i in $(seq 1 20); do
        if curl -sf http://localhost:11434/api/tags &>/dev/null; then break; fi
        sleep 1
    done
else
    info "ollama already running"
fi

info "Pulling model: $OLLAMA_MODEL"
ollama pull "$OLLAMA_MODEL"

# ── 4. whisper.cpp ────────────────────────────────────────────────────────────
step "whisper.cpp"
if [ ! -d "whisper.cpp" ]; then
    info "Cloning whisper.cpp …"
    git clone https://github.com/ggerganov/whisper.cpp --depth 1
fi

info "Building whisper.cpp (this takes ~5 min on Pi 5) …"
cmake -S whisper.cpp -B whisper.cpp/build \
    -DWHISPER_BUILD_EXAMPLES=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF          # aarch64: don't require AVX etc.
cmake --build whisper.cpp/build --config Release -j"$(nproc)"

info "Downloading whisper model: $WHISPER_MODEL"
if [ ! -f "whisper.cpp/models/ggml-${WHISPER_MODEL}.bin" ]; then
    bash whisper.cpp/models/download-ggml-model.sh "$WHISPER_MODEL"
else
    info "Model already present"
fi

# ── 5. piper ──────────────────────────────────────────────────────────────────
step "piper TTS"
if [ ! -d "piper" ]; then
    info "Downloading piper binary …"
    mkdir piper
    PIPER_TGZ="piper_linux_aarch64.tar.gz"
    wget -q --show-progress \
        "https://github.com/rhasspy/piper/releases/download/${PIPER_RELEASE}/${PIPER_TGZ}" \
        -O "/tmp/${PIPER_TGZ}"
    # Tarball contains a top-level 'piper/' folder — strip it
    tar xzf "/tmp/${PIPER_TGZ}" -C piper --strip-components=1
    rm "/tmp/${PIPER_TGZ}"
    chmod +x piper/piper
else
    info "piper directory already exists"
fi

if [ ! -f "piper/${PIPER_VOICE_NAME}.onnx" ]; then
    info "Downloading voice model: $PIPER_VOICE_NAME …"
    wget -q --show-progress \
        "${PIPER_HF_BASE}/${PIPER_VOICE_NAME}.onnx" \
        -O "piper/${PIPER_VOICE_NAME}.onnx"
    wget -q \
        "${PIPER_HF_BASE}/${PIPER_VOICE_NAME}.onnx.json" \
        -O "piper/${PIPER_VOICE_NAME}.onnx.json"
else
    info "Voice model already present"
fi

# ── 6. Python environment ─────────────────────────────────────────────────────
step "Python environment (uv)"
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
uv sync

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo "Run the voice assistant with:"
echo "    uv run python main.py"
echo ""
echo "Tip: make sure a microphone and speaker are connected."
echo "     Check audio devices with:  python3 -c \"import sounddevice; print(sounddevice.query_devices())\""
