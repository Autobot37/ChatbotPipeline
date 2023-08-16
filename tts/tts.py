import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from IPython.display import Audio
import numpy as np
from bark import generate_audio, SAMPLE_RATE

class TTS:
    def __init__(self):
        pass
    
    def audio_array(self, script: str, speaker: str = "v2/en_speaker_9") -> np.ndarray:
        audio_array = generate_audio(script, history_prompt=speaker)
        return audio_array
    
    def audio(self, script: str) -> Audio:
        ar = self.audio_array(script)
        return Audio(ar, rate=SAMPLE_RATE)  
    

 