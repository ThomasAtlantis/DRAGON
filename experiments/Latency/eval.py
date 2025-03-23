import sys, os
sys.path.append(".")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# from dragon import generator
# from dragon import transceiver
# from dragon import decoder
# from dragon import dragon
# from dragon import aggregator
# generator.logging_level = "ERROR"
# transceiver.logging_level = "ERROR"
# decoder.logging_level = "ERROR"
# dragon.logging_level = "ERROR"
# aggregator.logging_level = "ERROR"

import time
# import argparse
from tqdm import tqdm

from dragon.config import DragonConfig
from dragon.dragon import Dragon
from dragon.utils.stable import seed_everything
from dragon.utils.configure import Field as F
from experiments.evaluator import Evaluator
seed_everything(42)

def get_prompt_dataset(total=10):
    # import os
    # os.environ["http_proxy"] = "http://192.168.1.115:7890"
    # os.environ["https_proxy"] = "http://192.168.1.115:7890"
    # os.environ["all_proxy"] = "socks5://192.168.1.115:7890"
    # from datasets import load_dataset
    # dataset = load_dataset("data-is-better-together/10k_prompts_ranked", split="train")
    # dataset.save_to_disk("data/10k_prompts_ranked")

    from datasets import load_from_disk
    dataset = load_from_disk("data/10k_prompts_ranked")
    
    prompts = []
    for item in dataset:
        if 100 <= len(item['prompt']) <= 300 and not any(char.isdigit() for char in item['prompt']):
            prompts.append(item['prompt'])
        if len(prompts) == total:
            break
    
    return prompts


class LatencyConfig(DragonConfig):
    class evaluator:
        output_dir = F(str, default="outputs/", help="Output directory")
        max_new_tokens = F(int, default=100, help="Maximum number of tokens to generate")
        prompt_template = F(str, default="Context: {context}.\nInstruction: {query}\nAnswer: ", help="Prompt template")

class LatencyEvaluator(Evaluator):

    def __init__(self, config: LatencyConfig):
        super().__init__(config, name="Latency")
        self.device = Dragon(config)
        self.prompt_template = config.evaluator.prompt_template
        self.max_new_tokens = config.evaluator.max_new_tokens
        self.dataset = get_prompt_dataset(2)

    def evaluate(self):
        while not self.device.ready_for_generation:
            time.sleep(0.1)
        for query in tqdm(self.dataset):
            output_txt = self.device.query(query, self.prompt_template, self.max_new_tokens)
            print(output_txt)
        self.device.shutdown()

if __name__ == "__main__":
    config = LatencyConfig()
    config.retriever.passages = "wikipedia[remote]"
    config.generator.model = "facebook/opt-1.3b"
    config.generator.s_sequence = 896
    config.retriever.s_context = 128
    config.retriever.n_docs = 4
    config.retriever.s_aggregate = 4
    config.sampler.do_sample = False
    config.trans.tx_port = 6000
    config.trans.rx_port = 5000
    
    # args = argparse.ArgumentParser()
    # args.add_argument('--rank', type=int, default=0, help='0 for cloud, 1 for device')
    # args.add_argument('--method', type=str, default='synchronized', help='speculative or synchronized')
    # args = args.parse_args()
    
    # config.trans.rank = args.rank
    # config.aggregator.mode = args.method
    if config.trans.rank == 0:
        config.trans.tx_port, config.trans.rx_port = config.trans.rx_port, config.trans.tx_port
        cloud = Dragon(config)
    else:
        evaluator = LatencyEvaluator(config)
        evaluator.evaluate()

