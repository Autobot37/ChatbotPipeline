import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pickle
import torch
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_dir = os.path.dirname(current_dir)
sys.path.append(pipeline_dir)

from llm.model import LLM 
import sounddevice as sd
from bark import SAMPLE_RATE

class QNA:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        with open('C:\\Users\\SHIVA SINGH\\Documents\\Pipeline\\embedding_map.pkl', 'rb') as embedding_file:
            self.embedding_map = pickle.load(embedding_file)
        
        with open('C:\\Users\\SHIVA SINGH\\Documents\\Pipeline\\answer_map.pkl', 'rb') as answer_file:
            self.answer_map = pickle.load(answer_file)
        

    def answer(self, query: str,file_path:str="C:\\Users\\SHIVA SINGH\\Documents\\Pipeline\\Data Set.xlsx"):
        query_embedding = self.model.encode([query], convert_to_tensor=True)[0].cpu().detach().numpy()
        
        answer = {"type":None,"value":None,"update":None}

        most_similar_query = None
        max_similarity = 0.25
        for q, embedding in self.embedding_map.items():
            similarity = util.pytorch_cos_sim(query_embedding, embedding).item()
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_query = q
        #print(max_similarity)
        if most_similar_query is None:
            llm = LLM()
            out = llm.answer(query)
            answer["type"] = llm
            answer["value"] = out
            answer["update"] = 0
            answer["query"] = query
        
        df = pd.read_excel(file_path)
        prod = df.loc[df["utterance"] == most_similar_query, "category"].values[0]
        if prod == "PRODUCT":
            words = query.split()
            answer["type"] = "product"
            answer["value"] = self.answer_map[most_similar_query]
            answer["update"] = 0

            #filters
            answer["color"] = None
            answer["size"] = None
            answer["price"] = None
            for word in words:
                category = self.classify_word(word)
                if category == "color":
                    answer["color"] = word
                elif category == "size":
                    answer["size"] = word.upper()
                elif category == "price":
                    answer["price"] = int(word)
            
            if answer["color"] is None and answer["size"] is None and  answer["price"] is None:
                ret = self.ask()
                if ret == "ok":
                    return answer
                else:
                    answer["value"] = None
                    return answer
            else:
                return answer
        
        answer["type"] = "normal"
        answer["update"] = 0
        answer["value"] = self.answer_map.get(most_similar_query, "No answer found")
        return answer

    def classify_word(self, word):
        color_list = ["red", "blue", "green", "yellow", "black", "white", "orange"]
        size_list = ["s", "m", "l", "xl"]
        
        if word.lower() in color_list:
            return "color"
        elif word.lower() in size_list:
            return "size"
        elif word.isnumeric() and int(word) > 100:
            return "price"
        else:
            return None


    def ask(self):
        pickle_path = "C:\\Users\\SHIVA SINGH\\Documents\\Pipeline\\audio_map.pkl"
        with open(pickle_path, 'rb') as file:
            audio_map = pickle.load(file)
        print("Do you want any filters")
        arr = audio_map["askfilters"]
        sd.play(arr, SAMPLE_RATE)
        sd.wait()
        arr = audio_map["yesno"]
        sd.play(arr,SAMPLE_RATE)
        sd.wait()
        conf = input("Type Y for yes and N for NO:")
        
        if conf == "Y":
            print("Say again with product and filters")
            arr = audio_map["sayagain"]
            sd.play(arr,SAMPLE_RATE)
            sd.wait()
            return "again"
        else:
            print("fine")
            return "ok"


# df = pd.read_excel('Data Set.xlsx')  
# qna = QNA()

# query = "how to login to the account"
# response = qna.answer(query)
# print(response)
