import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Load Excel sheet
excel_path = 'Data Set.xlsx'
df = pd.read_excel(excel_path)

# Assuming your text column is named 'text'
texts = df['utterance'].tolist()
texts = texts[::2]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.to('cpu')

embeddings = model.encode(texts, convert_to_tensor=True)
tsne = TSNE(n_components=2, random_state=42)

# Explicitly move the embeddings to CPU and convert to NumPy
embeddings_2d = tsne.fit_transform(embeddings.cpu().detach().numpy())

# Assuming your DataFrame has an 'intent' column
intents = df['intent'].tolist()
intents = intents[::2]

# Create a color map for unique intents
unique_intents = np.unique(intents)
colors = plt.cm.get_cmap('tab10', len(unique_intents))

# Create a dictionary to map intents to colors
intent_to_color = {intent: colors(i / len(unique_intents)) for i, intent in enumerate(unique_intents)}

# Assign colors to each data point based on intent
point_colors = [intent_to_color[intent] for intent in intents]

# Create the scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=point_colors, s=10)
plt.title('t-SNE Visualization of Sentence Embeddings')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

# Create a legend for intents
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=intent_to_color[intent], markersize=8, label=intent) for intent in unique_intents]
plt.legend(handles=handles, title='Intents', loc='upper right')

plt.show()