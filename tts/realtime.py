import speech_recognition as sr
import numpy as np
from faster_whisper import WhisperModel
import io
import soundfile as sf
from rich import print

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

def main():

    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)

    print("Transcribing audio...")
    text = transcribe_audio(audio)

    print("Transcription:")
    styled_text = f"[bold italic green]{text}[/bold italic green]"

    print(styled_text)

if __name__ == "__main__":
    main()
