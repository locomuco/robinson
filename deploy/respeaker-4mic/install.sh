#!/bin/sh
# ReSpeaker 4-Mic Array (AC108) on a Raspberry Pi 5 - run on the Pi, then reboot.
#
# RP1's I2S has no TDM, so the board's 4-slot frame is read by an RP1 PIO state
# machine instead (snd-pio-tdm, card "seeed4micpio", 4 x S32 @ 16 kHz).  The
# AC108 is configured by the seeed-voicecard driver (card "seeed4micvoicec"),
# whose clocks only run while a capture is open there - that is the
# seeed-codec-heartbeat service.
#
#   1. seeed-voicecard (HinTak fork, branch v6.14) + our kernel-6.18 patch
#   2. snd-pio-tdm (this directory; from avisual/respeaker-6mic-pi5, adapted)
#   3. overlay seeed-4mic-pi5, config.txt line, heartbeat service
#
# Out-of-tree modules: re-run this script after every kernel upgrade.
set -eu
cd "$(dirname "$0")"
HERE=$(pwd)
KR=$(uname -r)
HDR=/usr/src/linux-headers-$KR
HDR_COMMON=/usr/src/linux-headers-$(echo "$KR" | sed 's/-rpi-.*/-common-rpi/')
DEST=/lib/modules/$KR/updates/seeed
SRC=${SEEED_SRC:-$HOME/seeed-voicecard}

sudo apt-get install -y -qq build-essential device-tree-compiler "linux-headers-$KR"

echo "== 1/3 seeed-voicecard =="
if [ ! -d "$SRC/.git" ]; then
    git clone -q --depth 1 -b v6.14 https://github.com/HinTak/seeed-voicecard "$SRC"
fi
git -C "$SRC" checkout -q -- .
git -C "$SRC" apply "$HERE/seeed-voicecard-6.18.patch"
make -C "/lib/modules/$KR/build" M="$SRC" modules

echo "== 2/3 snd-pio-tdm =="
make -C "/lib/modules/$KR/build" M="$HERE" modules

sudo mkdir -p "$DEST"
sudo install -m 644 "$SRC/snd-soc-ac108.ko" "$SRC/snd-soc-seeed-voicecard.ko" \
    "$HERE/snd-pio-tdm.ko" "$DEST/"
sudo depmod -a

echo "== 3/3 overlay, config.txt, heartbeat =="
cpp -nostdinc -I "$HDR/include" -I "$HDR_COMMON/include" -undef -x assembler-with-cpp \
    seeed-4mic-pi5-overlay.dts | dtc -@ -q -H epapr -O dtb -o seeed-4mic-pi5.dtbo -
fdtoverlay -i /boot/firmware/bcm2712-rpi-5-b.dtb -o /dev/null seeed-4mic-pi5.dtbo
sudo install -m 644 seeed-4mic-pi5.dtbo /boot/firmware/overlays/
grep -qxF dtoverlay=seeed-4mic-pi5 /boot/firmware/config.txt \
    || echo dtoverlay=seeed-4mic-pi5 | sudo tee -a /boot/firmware/config.txt >/dev/null
sudo install -m 644 seeed-codec-heartbeat.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable -q seeed-codec-heartbeat

echo "Done. Reboot, then: arecord -D hw:CARD=seeed4micpio -c4 -r16000 -f S32_LE -d3 t.wav"
