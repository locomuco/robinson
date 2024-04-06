import os

import pvporcupine
from pvrecorder import PvRecorder

API_KEY = os.environ.get('PICOVOICE_API_KEY')
KEYWORD_PATH = 'porcupine_keyword/robinson_en.ppn'

def get_device_id_by_name(device_name):
    for index, device in enumerate(PvRecorder.get_available_devices()):
        if device_name in device:
            return index

def listen(device_name):
    porcupine = pvporcupine.create(
        access_key=API_KEY,
        keyword_paths=[KEYWORD_PATH],
        sensitivities=[0.5])

    index = get_device_id_by_name(device_name)

    recorder = PvRecorder(device_index=index, frame_length=porcupine.frame_length)
    recorder.start()

    while True:
        pcm = recorder.read()
        result = porcupine.process(pcm)
        if result >= 0:
            print("success")
            recorder.delete()
            porcupine.delete()
            return 1

if __name__ == "__main__":
    listen("seeed-4mic-voicecard")