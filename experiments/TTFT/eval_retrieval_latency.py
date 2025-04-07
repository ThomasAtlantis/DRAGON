import sys, os
sys.path.append(".")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import numpy as np
from tqdm import tqdm


from dragon.config import DragonConfig
from dragon.utils.meter import Statistics, TimeMeter
from dragon.baselines.centralized_rag import RagForGeneration
from dragon.utils.stable import seed_everything
from dragon.utils.configure import Field as F
from experiments.evaluator import Evaluator
seed_everything(42)

time_meter = TimeMeter()

def get_prompt_dataset(total=2):
    with open("datasets/prompts/prompts.json", "r") as f:
        prompts = json.load(f)[: total]
    return prompts

class TTFTConfig(DragonConfig):
    class evaluator:
        output_dir = F(str, default="outputs/", help="Output directory")
        max_new_tokens = F(int, default=100, help="Maximum number of tokens to generate")
        prompt_template = F(str, default="Context: {context}.\nInstruction: {query}\nAnswer: ", help="Prompt template")
        n_prompts = F(int, default=2, help="Number of prompts to evaluate")

class TTFTEvaluator(Evaluator):

    def __init__(self, config: TTFTConfig):
        super().__init__(config, name="TTFT")
        self.ready_for_generation = False
        self.prompt_template = config.evaluator.prompt_template
        self.max_new_tokens = config.evaluator.max_new_tokens
        self.dataset = get_prompt_dataset(config.evaluator.n_prompts)
        self.dragon_config = config
        self.rag = RagForGeneration(config)
        self.stats = Statistics()
    
    def evaluate(self):
        for i in range(10):
            for item in tqdm(self.dataset):
                self.stats.new_record()
                query = item["query"]
                query_ids = self.rag.generator.tokenizer.encode(query)
                with time_meter.timer("RetrievalLatency"):
                    context_input_ids, attention_mask, scores, doc_texts = self.rag._prepare_inputs_for_generation(
                        query_ids, [], self.prompt_template)
                self.stats.update(time_meter.timer("RetrievalLatency"))
        self.stats.dump("outputs/retrieval_latency.json")
        retrieval_latency_avg = np.mean([rec["RetrievalLatency"] for rec in self.stats.records])
        self.logger.info(f"Average retrieval latency: {retrieval_latency_avg:.4f} s")
        # local retrieval latency: 0.05ms

if __name__ == "__main__":
    config = TTFTConfig()
    config.parse_sys_args()
    config.generator.s_sequence = 1024
    config.retriever.s_context = 256
    config.sampler.do_sample = False
    evaluator = TTFTEvaluator(config)
    evaluator.evaluate()