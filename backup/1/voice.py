from kokoro import KPipeline

import soundfile as sf
import sounddevice as sd


print('Voice Module Staring...')  # Notify User Of Module Started
pipeline = KPipeline(lang_code='a')  # Initialize the KPipeline with a language code


def text_to_speech(text: str):
    generator = pipeline(text, voice='bm_george')

    for i, (_, _, audio) in enumerate(generator):
        # Play Audio Directly To Headphones
        sd.play(audio, samplerate=24000)
        sd.wait() # Wait for the sound to finish playing

        # Optionally Save The Audio To A File (Enable if needed)
        #sf.write(f'{i}.wav', audio, 24000) 

if __name__ == '__main__':
    # For standalone testing
    test_text = "Hello, this is a test."
    text_to_speech(test_text) 

