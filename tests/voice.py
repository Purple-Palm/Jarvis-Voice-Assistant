from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import sounddevice as sd
import torch

# Notify User Of Module Started
print('Voice Module Staring...')
pipeline = KPipeline(lang_code='a')
text = '''
Allow me to introduce myself. I am Jarvis. A virtual artificial intelligence. And I am here to assist you with the variety of tasks as best I can. Twenty four hours a day. Seven days a week. Importing all preferences from home interface. Systems now fully operational.
'''
generator = pipeline(text, voice='bm_george')

for i, (gs, ps, audio) in enumerate(generator):
    print(i, gs, ps)
    display(Audio(data=audio, rate=24000, autoplay=i==0))
    sd.play(audio, samplerate=24000)
    sd.wait()
    sf.write(f'{i}.wav', audio, 24000) 