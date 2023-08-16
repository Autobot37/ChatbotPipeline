from nanoGPT import GPT,GPTConfig
import tiktoken
import torch
enc = tiktoken.get_encoding("gpt2")

class LLM:
    def __init__(self, path="C:\\Users\\SHIVA SINGH\\Documents\\Pipeline\\llm\\ckpt.pt") -> None:
        self.model = GPT(GPTConfig)
        self.model.load_state_dict(torch.load(path))
    def answer(self,query:str) -> str:
        idx = torch.tensor(enc.encode_ordinary(query), dtype=torch.long)
        idx = idx.unsqueeze(0)
        out = self.model.generate(idx, max_new_tokens=25)
        return enc.decode(out[0].tolist())



llm = LLM()
ans = llm.answer("hi")
print(ans)