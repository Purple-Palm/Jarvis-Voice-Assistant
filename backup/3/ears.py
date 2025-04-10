import asyncio
import os
import numpy as np
import whisper
import torch
import speech_recognition as sr
from queue import Queue
from datetime import datetime, timedelta


# Import functions from llm.py and voice.py
from llm import get_llm_response
from voice import text_to_speech

async def process_audio(audio_model, data_queue, phrase_timeout):
    """Audio processing and output."""
    last_phrase_time = None

    while True:
        now = datetime.utcnow()

        # Queue check
        if not data_queue.empty():
            # Ends the current phrase if paused
            if last_phrase_time and now - last_phrase_time > timedelta(seconds=phrase_timeout):
                print("\n--- End of sentence ---\n", flush=True)

            last_phrase_time = now

            # Collecting data from the queue
            audio_data = b''.join(data_queue.queue)
            data_queue.queue.clear()

            # Audio conversion for whisper
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Speech recognition in English ONLY
            result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available(), language="en")
            text = result['text'].strip()

            # Output
            if text:
                print(f"User: {text}", flush=True)
                # Get LLM response (run in executor to avoid blocking async loop)
                loop = asyncio.get_event_loop()
                llm_response = await loop.run_in_executor(None, get_llm_response, text)
                print(f"Jarvis: {llm_response}", flush=True)
                # Convert response to speech
                await loop.run_in_executor(None, text_to_speech, llm_response)
        else:
            await asyncio.sleep(0.1)  # Delay


async def main():
    # Parameters
    energy_threshold = 1000  # Volume threshold for recording
    record_timeout = 1.5     # Recording duration in seconds
    phrase_timeout = 2.0     # Pause between phrases to end the current one
    model_name = "large-v3"    # Model Whisper
    sample_rate = 42000      # Microphone sampling frequency

    # Queue for audio data
    data_queue = Queue()

    # Loading Whisper
    print("Loading the model...")
    audio_model = whisper.load_model(model_name, device="cpu") # Use "cuda" for GPU


    # Microphone setup
    recorder = sr.Recognizer()
    recorder.energy_threshold = energy_threshold
    recorder.dynamic_energy_threshold = False
    source = sr.Microphone(sample_rate=sample_rate)

    with source:
        recorder.adjust_for_ambient_noise(source)
    print("The microphone is set up. Getting started.\n")

    def record_callback(_, audio: sr.AudioData) -> None:
        """Background audio recording"""
        data_queue.put(audio.get_raw_data())

    # Background audio recording
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Processing
    await process_audio(audio_model, data_queue, phrase_timeout)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nThe program is complete.\n")