import sounddevice as sd

# List all audio devices
devices = sd.query_devices()
for i, device in enumerate(devices):
    if device['max_input_channels'] > 0:
        print(f"Input Device ID {i}: {device['name']}")