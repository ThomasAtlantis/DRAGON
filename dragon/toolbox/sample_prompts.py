# import os
# os.environ["http_proxy"] = "http://192.168.1.115:7890"
# os.environ["https_proxy"] = "http://192.168.1.115:7890"
# os.environ["all_proxy"] = "socks5://192.168.1.115:7890"
# from datasets import load_dataset
# dataset = load_dataset("data-is-better-together/10k_prompts_ranked", split="train")
# dataset.save_to_disk("data/10k_prompts_ranked")

import json
from datasets import load_from_disk
from dragon.utils.stable import seed_everything
from dragon.utils.configure import Configure, Field as F

class Config(Configure):
    seed = F(int, default=42, help="Random seed")
    total = F(int, default=10, help="Total number of prompts to sample")
    output_file = F(str, default="prompts.json", help="Output file path")
config = Config()
config.parse_sys_args()

seed_everything(config.seed)
dataset = load_from_disk("data/10k_prompts_ranked")
prompts = []
for item in dataset.shuffle():
    if 100 <= len(item['prompt']) <= 300 and not any(char.isdigit() for char in item['prompt']):
        prompts.append({
            "query": item['prompt'],
        })
    if len(prompts) == config.total:
        break

with open(config.output_file, "w") as f:
    json.dump(prompts, f, indent=4)
