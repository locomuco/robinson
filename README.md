# robinson

robinson is a retro robot built based on the idea of 
https://hackaday.io/project/189041-a-workbench-companion-from-an-amazon-echo-dot

Instead of using Alexa as a Base device, the Robot is running on a raspberry PI

# Getting started
```
sudo apt-get install portaudio19-dev libatlas-base-dev


curl -sSf https://rye-up.com/get | bash
echo 'source "$HOME/.rye/env"' >> ~/.bashrc
source ~/.bashrc

rye sync
```

# Wake Word Detection

Picovoice requires an online account to run on-device wake word detection.
Add the 
```
export PICOVOICE_API_KEY="<AccessKey>"
```