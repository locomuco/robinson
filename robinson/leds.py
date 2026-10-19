"""The ReSpeaker 4-Mic Array's ring of 12 APA102 LEDs.

They show where Robinson hears speech (a spot on the ring, white when it is
not addressed, blue when it is listening to you) and what it is doing
(thinking: a purple comet; speaking: breathing cyan; follow-up window: a
faint blue ring).

Hardware: SPI0 (/dev/spidev0.0, needs dtparam=spi=on), LED power switched
by GPIO5. Without them the class does nothing.
"""

import fcntl
import logging
import math
import os
import struct
import threading
import time

log = logging.getLogger("robinson.leds")

N = 12
_SPI_IOC_WR_MAX_SPEED_HZ = 0x40046B04

WHITE = (255, 255, 255)
BLUE = (0, 90, 255)
PURPLE = (150, 0, 255)
CYAN = (0, 200, 160)


class Leds(threading.Thread):
    def __init__(self, state_fn, led0_deg: float = 0.0, mirror: bool = False,
                 brightness: float = 0.3, spi: str = "/dev/spidev0.0", power_gpio: int = 5):
        """state_fn() -> (eye state, awake): the robot's state, read every frame."""
        super().__init__(name="leds", daemon=True)
        self.state_fn = state_fn
        self.led0_deg, self.mirror, self.brightness = led0_deg, mirror, brightness
        self._az: float | None = None       # array frame
        self._az_t = 0.0
        self._stop = threading.Event()
        self.fd = None
        self._gpio = None
        try:
            self.fd = os.open(spi, os.O_WRONLY)
            fcntl.ioctl(self.fd, _SPI_IOC_WR_MAX_SPEED_HZ, struct.pack("I", 8_000_000))
            import lgpio
            self._gpio = lgpio.gpiochip_open(0)
            lgpio.gpio_claim_output(self._gpio, power_gpio, 1)
            log.info("LED ring on %s", spi)
        except Exception as e:  # noqa: BLE001
            log.info("no LED ring (%s)", e)
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None

    @property
    def available(self) -> bool:
        return self.fd is not None

    def direction(self, az_array: float, _strength: float = 0.0) -> None:
        self._az, self._az_t = az_array, time.monotonic()

    def stop(self) -> None:
        self._stop.set()

    # ── rendering ────────────────────────────────────────────────────────────
    def _led_angle(self, i: int) -> float:
        return (self.led0_deg + (-30 if self.mirror else 30) * i) % 360

    def _frame(self, now: float) -> list[tuple[float, float, float]]:
        state, awake = self.state_fn()
        px = [(0.0, 0.0, 0.0)] * N
        if state == "sleep":
            return px
        if state == "thinking":
            head = (now * 1.5 * N) % N
            for i in range(N):
                k = 0.85 ** ((head - i) % N)
                px[i] = tuple(c * k for c in PURPLE)
            return px
        if state == "speaking":
            k = 0.35 + 0.25 * math.sin(now * 4)
            return [tuple(c * k for c in CYAN)] * N
        if awake:                            # follow-up window: no wake word needed
            px = [tuple(c * 0.08 for c in BLUE)] * N
        age = now - self._az_t
        if self._az is not None and age < 2.0:
            color = BLUE if awake else WHITE
            fade = 1.0 if age < 0.6 else max(0.0, 1 - (age - 0.6) / 1.4)
            for i in range(N):
                d = abs((self._led_angle(i) - self._az + 180) % 360 - 180)
                k = math.exp(-(d / 22) ** 2) * fade * (1.0 if awake else 0.5)
                if k > 0.02:
                    px[i] = tuple(max(p, c * k) for p, c in zip(px[i], color))
        return px

    def _show(self, px) -> None:
        data = bytearray(4)
        b = 0xE0 | max(1, min(31, int(31 * self.brightness)))
        for r, g, bl in px:
            data += bytes((b, int(bl), int(g), int(r)))
        data += b"\xff" * 4
        os.write(self.fd, bytes(data))

    def run(self) -> None:
        if not self.available:
            return
        while not self._stop.is_set():
            try:
                self._show(self._frame(time.monotonic()))
            except Exception as e:  # noqa: BLE001
                log.warning("LED update failed: %s", e)
                return
            self._stop.wait(1 / 30)
        self._show([(0, 0, 0)] * N)
