import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import transformers


tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/contriever")
retriever = transformers.AutoModel.from_pretrained("facebook/contriever")
tokenizer.save_pretrained("./models/contriever")
retriever.save_pretrained("./models/contriever")