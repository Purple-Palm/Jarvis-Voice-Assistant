# Wake word detection script using openwakeword
import pyaudio
import numpy as np
from openwakeword.model import Model
import asyncio

async def listen_for_wake_word():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1280
    WAKEWORD_MODEL = "hey_jarvis"

    audio = pyaudio.PyAudio()
    mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    model = Model(wakeword_models=[WAKEWORD_MODEL], inference_framework="tflite")

    print("\nListening for 'hey_jarvis' wakeword...")

    while True:
        audio_data = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)
        prediction = model.predict(audio_data)

        scores = list(model.prediction_buffer[WAKEWORD_MODEL])
        if scores[-1] > 0.4:
            print("Wakeword Detected!")
            break

    mic_stream.stop_stream()
    mic_stream.close()
    audio.terminate()

    print("Wake word triggered!")