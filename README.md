# ChatbotPipeline
# Without OpenAI

It is a Python module for full conversation chatbot.

## What
I had enough of hardware usage of LLMS, and slowness and then wrong answers sometimes.So, i think generality is difficult to achieve.
and i dint wanted to use openai api.

## So What I Did
Actually there was war between computation and storage, surely Gpt3/4 is ultimate beast, so we could use it to make Dataset of plausible querys, and answers, and now you know what is this.I just have to match given query to one of the query.

# But
What i liked is there are also limited conditions where you have to speak, that is dataset answers + some descriptions and guides.
So i also stored that voices data and sample_rate.

# Dataset
here if we are storing so much then we have to find efficient storage,for now i am using maps[will upgrade soon] also i didn't used vector databases, for now.

# So, What is the Result
That it is blazing Fast even on cpu.

   
## Installation
```bash
git clone https://github.com/Autobot37/ChatbotPipeline
cd ChatbotPipeline
pip install -r requirements.txt

cd pipeline
python whole.py
```
## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.
