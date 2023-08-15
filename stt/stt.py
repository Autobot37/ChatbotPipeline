#pip install fast_whisper
from faster_whisper import WhisperModel
import torch

class STT:
    def __init__(self,model_size:str="base") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if(self.device == "cuda"):
            print("using cuda whisper")
            self.model = WhisperModel(model_size, device="cuda", compute_type="float32")
        else:
            print("using slow cpu whisper")
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        
        #self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    def transcribe(self,path:str,with_stamps:bool=False) -> str:
        segments, info = self.model.transcribe(path, beam_size=5)
        text = ""
        for segment in segments:
            text += segment.text

        return text

# stt = STT()
# stt.transcribe("Text-to-Speech_13-Aug-2023_18-26.mp3")

print(torch.cuda.is_available())