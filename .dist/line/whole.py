from stt import STT
from qa import QNA
import pandas as pd
import pickle
from bark import SAMPLE_RATE
from IPython.display import Audio

####STT#######
stt = STT()
audio_path = "test (1).mp3"
query = stt.transcribe(audio_path)

#####Q -> A ########
qa = QNA()
file_path = "Data Set.xlsx"
df = pd.read_excel(file_path)
answer = qa.answer(query, df)

#######TO AUDIO ############
pickle_path = "audio_map.pkl"
with open(pickle_path, 'rb') as file:
    audio_map = pickle.load(file)

answer_audio_array = audio_map[answer]

Audio(answer_audio_array, SAMPLE_RATE)
