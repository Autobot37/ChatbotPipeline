import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_dir = os.path.dirname(current_dir)
sys.path.append(pipeline_dir)
  
import pandas as pd
import pickle
from bark import SAMPLE_RATE
import sounddevice as sd
from load import stt,qa
####STT#######
audio_path = "Text-to-Speech_13-Aug-2023_18-26.mp3"
query = stt.transcribe(audio_path)
print(query)

#####Q -> A ########
file_path = "Data Set.xlsx"
df = pd.read_excel(file_path)
answer = qa.answer(query)
print(answer)

#######TO AUDIO ############
pickle_path = "audio_map.pkl"
with open(pickle_path, 'rb') as file:
    audio_map = pickle.load(file)
print("pickle loaded")
answer_audio_array = audio_map[answer]

if answer_audio_array is not None:
    sd.play(answer_audio_array, SAMPLE_RATE)
    sd.wait() 
else:
    print("We will reply shortly")
