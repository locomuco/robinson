"""Robinson's camera: the ESP-EYE on USB serial (firmware in esp-eye/).

A thread reads the JPEG stream (QVGA, ~10 fps), finds faces with OpenCV's
YuNet and reports the biggest one as a direction (on_face callback), so the
eyes can follow it. snapshot() fetches one larger frame (800x600) for the
LLM ("Was siehst du?").
"""

import glob
import logging
import struct
import threading
import time

import numpy as np

log = logging.getLogger("robinson.camera")

BAUD = 921_600
MAGIC = b"RBNF"


def find_port() -> str | None:
    ports = glob.glob("/dev/serial/by-id/*CP210*")      # the ESP-EYE's USB-UART bridge
    return ports[0] if ports else None


class EspEye(threading.Thread):
    def __init__(self, port: str, face_model: str, fov_deg: float = 56.0, mirror: bool = False):
        super().__init__(name="camera", daemon=True)
        import cv2
        import serial
        self.cv2 = cv2
        self.ser = serial.Serial(port, 115_200, timeout=0.2)
        self.fov, self.mirror = fov_deg, mirror
        self.detector = cv2.FaceDetectorYN.create(face_model, "", (320, 240), 0.8)
        self.on_face = None              # callback(azimuth, elevation, size): + = left / up
        self.face_t = 0.0                # when a face was last seen
        self.faces = 0                   # how many in the last frame
        self._hi_req = threading.Event()
        self._hi_done = threading.Event()
        self._hi: bytes | None = None
        self._stop = threading.Event()

    # ── public ───────────────────────────────────────────────────────────────
    def face_age(self) -> float:
        return time.monotonic() - self.face_t

    def snapshot(self, timeout: float = 3.0) -> bytes | None:
        """One 800x600 JPEG, or None."""
        self._hi, self._hi_done = None, threading.Event()
        self._hi_req.set()
        self._hi_done.wait(timeout)
        return self._hi

    def stop(self) -> None:
        self._stop.set()

    # ── stream ───────────────────────────────────────────────────────────────
    def _boot(self) -> None:
        """Reset the ESP-EYE (it starts sending frames 2 s later at BAUD)."""
        self.ser.baudrate = 115_200
        self.ser.dtr, self.ser.rts = False, True
        time.sleep(0.1)
        self.ser.rts = False
        time.sleep(2.5)
        self.ser.baudrate = BAUD
        self.ser.reset_input_buffer()
        self.ser.write(b"q")

    def _frames(self):
        buf = b""
        last = time.monotonic()
        while not self._stop.is_set():
            if self._hi_req.is_set():
                self._hi_req.clear()
                self.ser.write(b"h")
            buf += self.ser.read(max(1, self.ser.in_waiting))
            if time.monotonic() - last > 5:      # ESP-EYE silent: reset it
                log.warning("camera silent - resetting the ESP-EYE")
                self._boot()
                buf, last = b"", time.monotonic()
                continue
            while (i := buf.find(MAGIC)) >= 0 and len(buf) >= i + 12:
                n, w, h = struct.unpack_from("<IHH", buf, i + 4)
                if n > 200_000:                  # garbage: skip the false magic
                    buf = buf[i + 4:]
                    continue
                if len(buf) < i + 12 + n:
                    break
                jpeg, buf = buf[i + 12:i + 12 + n], buf[i + 12 + n:]
                last = time.monotonic()
                yield w, h, jpeg
            if len(buf) > 400_000:
                buf = buf[-20_000:]

    def run(self) -> None:
        try:
            self._boot()
            log.info("camera (ESP-EYE) on %s", self.ser.port)
            for w, h, jpeg in self._frames():
                if w > 320:
                    self._hi = jpeg
                    self._hi_done.set()
                    continue
                self._detect(jpeg)
        except Exception as e:  # noqa: BLE001
            log.warning("camera stopped: %s", e)

    def _detect(self, jpeg: bytes) -> None:
        cv2 = self.cv2
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img)
        n = 0 if faces is None else len(faces)
        if (n > 0) != (self.faces > 0) and (n == 0 or self.face_age() > 3):
            log.info("face %s", f"seen ({n})" if n else "gone")
        self.faces = n
        if not n:
            return
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])[:4]
        cx, cy = (x + fw / 2) / w - 0.5, (y + fh / 2) / h - 0.5
        az = (cx if self.mirror else -cx) * self.fov       # image left = Robinson's right
        el = -cy * self.fov * h / w
        self.face_t = time.monotonic()
        if self.on_face:
            self.on_face(az, el, fw / w)
