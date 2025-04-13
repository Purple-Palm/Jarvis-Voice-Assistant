import asyncio
import numpy as np
import torch
import speech_recognition as sr
import webrtcvad
from queue import Queue
from datetime import datetime, timedelta
from transformers import pipeline

# Import functions from llm.py and voice.py
from llm import get_llm_response
from voice import text_to_speech

# Configuration ---------------------------------------------------
VAD_AGGRESSIVENESS = 3  # Maximum filtering (0-3)
MIN_AUDIO_LENGTH = 0.5  # Seconds of speech to process
ENERGY_THRESHOLD = 4000  # Fixed noise floor
# ----------------------------------------------------------------

vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
FRAME_SIZE = 160  # 10ms frames at 16kHz

async def process_audio(asr_pipe, data_queue, phrase_timeout):
    last_phrase_time = None
    audio_buffer = bytearray()

    while True:
        now = datetime.utcnow()
        
        # Buffer incoming audio
        while not data_queue.empty():
            audio_buffer.extend(data_queue.get_nowait())

        if len(audio_buffer) > 0:
            # Convert to numpy array
            audio_int16 = np.frombuffer(audio_buffer, dtype=np.int16)
            
            # VAD Validation -------------------------------
            if len(audio_int16) < FRAME_SIZE * 10:  # <100ms
                audio_buffer = bytearray()  # Reinitialize instead of clearing
                continue

            # Trim to frame multiples
            audio_int16 = audio_int16[:len(audio_int16) // FRAME_SIZE * FRAME_SIZE]
            
            # Check speech probability
            speech_frames = [
                vad.is_speech(
                    audio_int16[i:i+FRAME_SIZE].tobytes(),
                    sample_rate=16000
                )
                for i in range(0, len(audio_int16), FRAME_SIZE)
            ]
            speech_ratio = sum(speech_frames) / len(speech_frames)
            
            if speech_ratio < 0.7:  # Require 70% speech frames
                audio_buffer = bytearray()  # Reinitialize instead of clearing
                continue
            # ----------------------------------------------

            # Convert for Whisper
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            # Whisper Processing with Anti-Hallucination
            try:
                result = asr_pipe(
                    {"raw": audio_float32, "sampling_rate": 16000},
                    generate_kwargs={
                        "language": "en",
                        "task": "transcribe",
                        "temperature": 0.0,  # Disable randomness
                        #"suppression_tokens": [1, 2, 7],  # Suppress common garbage
                    }
                )
                text = result['text'].strip()
            except Exception as e:
                print(f"ASR Error: {e}")
                continue

            # Validate output
            if len(text) < 2:  # Ignore single-letter outputs
                continue
                
            print(f"User: {text}", flush=True)
        
            # LLM Processing
            try:
                loop = asyncio.get_event_loop()
                llm_response = await loop.run_in_executor(None, get_llm_response, text)
                print(f"Jarvis: {llm_response}", flush=True)
                await loop.run_in_executor(None, text_to_speech, llm_response)
            except Exception as e:
                print(f"Response Error: {e}")

            audio_buffer = bytearray()  # Reinitialize instead of clearing
            last_phrase_time = now

        else:
            await asyncio.sleep(0.1)

async def main():
    # Force disable dynamic energy threshold
    recorder = sr.Recognizer()
    recorder.energy_threshold = ENERGY_THRESHOLD
    recorder.dynamic_energy_threshold = False  # Critical fix
    
    # Whisper initialization
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16,
    )

    # Microphone setup
    source = sr.Microphone(sample_rate=16000)  # Remove 'with' statement
    print(f"Microphone detected: {source}")
    print("Calibrating microphone...")
    with source:
        recorder.adjust_for_ambient_noise(source, duration=2)
    print(f"Final energy threshold: {recorder.energy_threshold}")

    def callback(_, audio: sr.AudioData):
        data_queue.put(audio.get_raw_data())

    data_queue = Queue()
    recorder.listen_in_background(source, callback, phrase_time_limit=1.5)
    
    await process_audio(asr_pipe, data_queue, phrase_timeout=2.0)

if __name__ == "__main__":
    asyncio.run(main())