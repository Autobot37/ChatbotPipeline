import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pickle
import torch

class QNA:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        with open('embedding_map.pkl', 'rb') as embedding_file:
            self.embedding_map = pickle.load(embedding_file)
        
        with open('answer_map.pkl', 'rb') as answer_file:
            self.answer_map = pickle.load(answer_file)

    def answer(self, query: str) -> str:
        query_embedding = self.model.encode([query], convert_to_tensor=True)[0].cpu().detach().numpy()
        
        most_similar_query = None
        max_similarity = 0.0
        for q, embedding in self.embedding_map.items():
            similarity = util.pytorch_cos_sim(query_embedding, embedding).item()
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_query = q
        
        if most_similar_query is None:
            return "No match found"
        
        answer = self.answer_map.get(most_similar_query, "No answer found")
        return answer

df = pd.read_excel('Data Set.xlsx')  
qna = QNA()

query = "mens tshirt"
#response = qna.answer(query)
#print(response)
print(len(df["utterance"].tolist()))