from nanoGPT import GPTConfig, GPT
GPTConfig.n_layer = 4
GPTConfig.n_head = 4
GPTConfig.block_size = 32

import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning.pytorch as pl

import tiktoken 
enc = tiktoken.get_encoding("gpt2")

class Model(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.model = GPT(GPTConfig)
  def training_step(self, batch, batch_idx):
    self.model.train()
    x, y = batch
    out, loss = self.model(x,y)
    return loss
  
  def configure_optimizers(self):
    return self.model.configure_optimizers(1e-1,6e-4,(0.9, 0.95),self.device)
  
 
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

file_path = "C:\\Users\\SHIVA SINGH\\Documents\\Pipeline\\Data Set.xlsx"
df = pd.read_excel(file_path)

text_list = [f"{row['utterance']} {row['responses']}" for _, row in df.iterrows()]
text_list = [enc.encode_ordinary(l) for l in text_list]
idx_list = []
for ids in text_list:
  for id in ids:
    idx_list.append(id)

input_sequences = []
target_sequences = []

for i in range(0,len(idx_list) - 32, 32):
    input_seq = idx_list[i:i+32]
    input_seq = torch.tensor(input_seq,device="cpu",dtype=torch.long)
    target_seq = idx_list[i+1:i+1+32]
    target_seq = torch.tensor(target_seq,device="cpu",dtype=torch.long)
    input_sequences.append(input_seq)
    target_sequences.append(target_seq)

model = Model()

class CustomDataset(Dataset):
    def __init__(self, input_sequences, target_sequences):
        self.input_sequences = input_sequences
        self.target_sequences = target_sequences

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        return self.input_sequences[idx], self.target_sequences[idx]

train_size = int(0.8 * len(input_sequences))
val_size = len(input_sequences) - train_size

train_dataset = CustomDataset(input_sequences[:train_size], target_sequences[:train_size])
val_dataset = CustomDataset(input_sequences[train_size:], target_sequences[train_size:])

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

trainer = pl.Trainer(strategy="auto", enable_progress_bar=True,max_epochs=1)
trainer.fit(model=model,train_dataloaders = train_loader,val_dataloaders=val_loader)


checkpoint_path = "C:\\Users\\SHIVA SINGH\\Documents\\Pipeline\\llm\\ckpt.pt"
torch.save(model.model.state_dict(), checkpoint_path)