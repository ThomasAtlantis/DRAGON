import sys, os
sys.path.append(".")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import numpy as np
from tqdm import tqdm


from dragon.config import DragonConfig
from dragon.utils.meter import Statistics, TimeMeter
from dragon.baselines.centralized_rag import RagForGeneration, group_docs
from dragon.utils.stable import seed_everything
from dragon.utils.configure import Field as F
from experiments.evaluator import Evaluator
seed_everything(42)

time_meter = TimeMeter()

def get_prompt_dataset(total=2):
    with open("prompts.json", "r") as f:
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
        for item in tqdm(self.dataset):
            self.stats.new_record()
            query = item["query"]
            query_ids = self.rag.generator.tokenizer.encode(query)
        
            query = self.rag.generator.tokenizer.decode(query_ids)
            docs, scores = self.rag.retriever.retrieve_passages([query])[0]
            doc_texts = [doc["text"] for doc in docs]
            doc_texts, scores = group_docs(doc_texts, scores, self.rag.aggregate_size)
            
            doc_texts = [
                self.rag.generator.tokenizer.decode(self.rag.generator.tokenizer.encode(doc_text)[: self.rag.context_size])
                for doc_text in doc_texts
            ]
            prefix = self.prompt_template.split('\n')[0].format(context=doc_texts[0])
            postfix = '\n'.join(self.prompt_template.split('\n')[1: ]).format(query=query)
            prefix_ids = self.rag.generator.tokenizer.encode(prefix)
            postfix_ids = self.rag.generator.tokenizer.encode(postfix)
            prefix_proportion = len(prefix_ids) / (len(prefix_ids) + len(postfix_ids))
            self.stats.update(name="PrefixProportion", stat=prefix_proportion)
        self.stats.dump("outputs/prefix_proportion.json")
        self.logger.info("Prefix proportion saved to `outputs/prefix_proportion.json`")

if __name__ == "__main__":
    config = TTFTConfig()
    config.parse_sys_args()
    config.generator.s_sequence = 1024
    config.retriever.s_context = 256
    config.sampler.do_sample = False
    evaluator = TTFTEvaluator(config)
    evaluator.evaluate()