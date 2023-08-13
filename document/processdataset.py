import pandas as pd 
from sentence_transformers import SentenceTransformer, util
import pickle

excel_path = "Data Set.xlsx"
df = pd.read_excel(excel_path)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.to("cuda")

def cosine_similarity(embedding1, embedding2):
    return util.pytorch_cos_sim(embedding1, embedding2).item()

embedding_map = {}
answer_map = {}

for index, row in df.iterrows():
    query = row["utterance"]
    answer = row["responses"]

    q_embed = model.encode([query])[0]

    if query in embedding_map:
        continue

    thresh = 0.9
    found = False
    for prev_query, prev_embedding in embedding_map.items():
        sim = cosine_similarity(q_embed, prev_embedding)
        if sim > thresh:
            found = True
            break
    
    if not found:
        embedding_map[query] = q_embed
        answer_map[query] = answer


# Save maps using pickle
with open('embedding_map.pkl', 'wb') as embedding_file:
    pickle.dump(embedding_map, embedding_file)

with open('answer_map.pkl', 'wb') as answer_file:
    pickle.dump(answer_map, answer_file)

print("Maps saved as pickle files.")
