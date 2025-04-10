# main.py
import asyncio
import whisper
import torch
import numpy as np
from queue import Queue
from datetime import datetime, timedelta

from wake_word import listen_for_wake_word
from llm import get_llm_response
from voice import text_to_speech

import speech_recognition as sr

async def record_and_transcribe(model_name="tiny", sample_rate=16000, phrase_timeout=2.0):
    """Record a full sentence after wake word is detected."""
    data_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = 1000
    recorder.dynamic_energy_threshold = False
    source = sr.Microphone(sample_rate=sample_rate)

    with source:
        recorder.adjust_for_ambient_noise(source)
    print("Microphone ready for recording...")

    def callback(_, audio: sr.AudioData) -> None:
        data_queue.put(audio.get_raw_data())

    stop_recording = recorder.listen_in_background(source, callback, phrase_time_limit=1.5)
    print("Whisper listening for sentence...")

    last_phrase_time = datetime.utcnow()
    audio_model = whisper.load_model(model_name)

    while True:
        now = datetime.utcnow()
        if not data_queue.empty():
            last_phrase_time = now
            audio_data = b''.join(data_queue.queue)
            data_queue.queue.clear()

            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available(), language="en")
            text = result['text'].strip()

            if text:
                print(f"User: {text}")
                stop_recording(wait_for_stop=False)
                return text

        elif (now - last_phrase_time).total_seconds() > phrase_timeout:
            stop_recording(wait_for_stop=False)
            print("No sentence detected, back to idle.")
            return None

        await asyncio.sleep(0.1)


async def main():
    while True:
        print("Listening for wake word 'Jarvis'...")
        await listen_for_wake_word()

        print("Wake word detected! Start processing...")
        spoken_text = await record_and_transcribe()

        if spoken_text:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, get_llm_response, spoken_text)
            print(f"Jarvis: {response}")
            await loop.run_in_executor(None, text_to_speech, response)
        else:
            print("No speech detected after wake word.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down Jarvis.")
