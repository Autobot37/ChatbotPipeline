from tts import TTS
import pandas as pd
import numpy as np
import pickle
from bark import SAMPLE_RATE
tts = TTS()

def text_to_audio_array(text):
    audio_array = tts.audio_array(script = text, speaker="v2/en_speaker_9")
    return audio_array

input_excel_path = 'Data Set.xlsx'  
df = pd.read_excel(input_excel_path)

audio_map = {}

for index, row in df.iterrows():
    response = row['responses']
    
    if response not in audio_map:
        audio_data = text_to_audio_array(response)
        audio_map[response] = audio_data
   

####some custom statemts####
askfilters = "Do you want any filters"
yesno = "Type Y for yes and N for NO"
sayagain = "Say again with product and filters"
#############
data = text_to_audio_array(askfilters)
audio_map["askfilters"] = data
data = text_to_audio_array(yesno)
audio_map["yesno"] = data
data = text_to_audio_array(sayagain)
audio_map["sayagain"] = data
################

audio_map_path = 'audio_map.pkl'
with open(audio_map_path, 'wb') as f:
    pickle.dump(audio_map, f)

print("Audio map saved using pickle.")

