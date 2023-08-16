class Dataset:
    def __init__(self,file_path):
        pass

    def updateanswer(self,query, answer):
        ans_path = "answer_map.pkl"
        with open(pickle_path, 'rb') as file:
            ans_map = pickle.load(file)
        ans_map[query] = answer
    
    def updateaudio(self,answer):
        pickle_path = "audio_map.pkl"
        with open(pickle_path, 'rb') as file:
            audio_map = pickle.load(file)
        from tts.tts import TTS
        model = TTS()
        array = model.audio_array(answer)
        audio_map[answer] = array
    
        
        