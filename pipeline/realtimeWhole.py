import speech_recognition as sr
import numpy as np
from faster_whisper import WhisperModel
import io
import soundfile as sf
from rich import print
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_dir = os.path.dirname(current_dir)
sys.path.append(pipeline_dir)
  
import pandas as pd
import pickle
from bark import SAMPLE_RATE
import sounddevice as sd
from qa.qna import QNA


def transcribe_audio(audio_data):
    stt = WhisperModel("small", device="cpu", compute_type="float32")
    wav_bytes = audio_data.get_wav_data(convert_rate=16000)
    wav_stream = io.BytesIO(wav_bytes)
    audio_array, sampling_rate = sf.read(wav_stream)
    audio_array = audio_array.astype(np.float32)
    segments, info = stt.transcribe(audio_array, beam_size=5)
    text = ""
    for segment in segments:
        text += segment.text

    return text

qa = QNA()

def main():

    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)

    print("Transcribing audio...")
    query = transcribe_audio(audio)

    print("Transcription:")
    styled_text = f"[bold italic green]{query}[/bold italic green]"
    some = "Hey!, Your query is:"
    print(f"[bold italic red] {some} [/bold italic red]")
    print(styled_text)

    file_path = "C:\\Users\\SHIVA SINGH\\Documents\\Pipeline\\Data Set.xlsx"
    df = pd.read_excel(file_path)
    answer = qa.answer(query)
    print(answer)

    #######TO AUDIO ############
    pickle_path = "C:\\Users\\SHIVA SINGH\\Documents\\Pipeline\\audio_map.pkl"
    with open(pickle_path, 'rb') as file:
        audio_map = pickle.load(file)
    answer_audio_array = audio_map[answer]

    if answer_audio_array is not None:
        sd.play(answer_audio_array, SAMPLE_RATE)
        sd.wait() 
    else:
        print("We will reply shortly")



if __name__ == "__main__":
    main()
