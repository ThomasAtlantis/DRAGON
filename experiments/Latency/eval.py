import json
from pathlib import Path
import sys, os
sys.path.append(".")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dragon import generator
from dragon import transceiver
from dragon import decoder
from dragon import dragon
from dragon import aggregator
generator.logging_level = "ERROR"
transceiver.logging_level = "ERROR"
decoder.logging_level = "ERROR"
dragon.logging_level = "ERROR"
aggregator.logging_level = "ERROR"

import time
# import argparse
from tqdm import tqdm

from dragon.config import DragonConfig
from dragon.dragon import Dragon
from dragon.utils.stable import seed_everything
from dragon.utils.configure import Field as F
from experiments.evaluator import Evaluator
seed_everything(42)

def get_prompt_dataset(total=2):
    with open("prompts.json", "r") as f:
        prompts = json.load(f)[: total]
    return prompts

class LatencyConfig(DragonConfig):
    class evaluator:
        output_dir = F(str, default="outputs/", help="Output directory")
        max_new_tokens = F(int, default=100, help="Maximum number of tokens to generate")
        prompt_template = F(str, default="Context: {context}.\nInstruction: {query}\nAnswer: ", help="Prompt template")
        n_prompts = F(int, default=2, help="Number of prompts to evaluate")

class LatencyEvaluator(Evaluator):

    def __init__(self, config: LatencyConfig):
        super().__init__(config, name="Latency")
        self.device = Dragon(config)
        self.prompt_template = config.evaluator.prompt_template
        self.max_new_tokens = config.evaluator.max_new_tokens
        self.dataset = get_prompt_dataset(config.evaluator.n_prompts)

    def evaluate(self):
        while not self.device.ready_for_generation:
            time.sleep(0.1)
        for item in tqdm(self.dataset):
            print(item['query'])
            output_txt = self.device.query(item['query'], self.prompt_template, self.max_new_tokens)
            print(output_txt)

        run_output_dir = Path(self.config.output_dir, self.run_id)
        run_output_dir.mkdir(parents=True, exist_ok=True)
        stats_file = run_output_dir / f"stats.json"
    
        from dragon.aggregator import stats as aggregator_stats
        (self.device.stats | aggregator_stats).dump(stats_file)
        self.logger.info(f"Stats saved to `{stats_file}`")
        self.device.shutdown()

if __name__ == "__main__":
    config = LatencyConfig()
    config.parse_sys_args()
    config.generator.s_sequence = 1024
    config.retriever.s_context = 256
    config.sampler.do_sample = False
    config.trans.tx_port = 6000
    config.trans.rx_port = 6001
    
    if config.trans.rank == 0:
        config.trans.tx_host = "192.168.1.115"
        config.trans.tx_port, config.trans.rx_port = config.trans.rx_port, config.trans.tx_port
        cloud = Dragon(config)
    else:
        config.trans.tx_host = "192.168.1.126"
        evaluator = LatencyEvaluator(config)
        evaluator.evaluate()

