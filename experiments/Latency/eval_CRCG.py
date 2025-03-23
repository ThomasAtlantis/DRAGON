from pathlib import Path
import sys, os

sys.path.append(".")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from tqdm import tqdm
from dragon.config import DragonConfig
from dragon.utils.stable import seed_everything
from dragon.utils.configure import Field as F
from experiments.evaluator import Evaluator
from dragon.baselines.centralized_rag import RagSequenceForGeneration, RagTokenForGeneration

seed_everything(42)


def get_prompt_dataset():
    with open("prompts.json", "r") as f:
        prompts = f.readlines()
    return prompts

class LatencyConfig(DragonConfig):
    class evaluator:
        output_dir = F(str, default="outputs/", help="Output directory")
        max_new_tokens = F(int, default=100, help="Maximum number of tokens to generate")
        prompt_template = F(str, default="Context: {context}.\nInstruction: {query}\nAnswer: ", help="Prompt template")
        method = F(str, default="seq", help="Method for evaluation: seq or tok")

class LatencyEvaluator(Evaluator):
    
    def __init__(self, config: LatencyConfig):
        super().__init__(config, name="Latency")
        if config.evaluator.method == "seq":
            self.rag = RagSequenceForGeneration(config)
        else:
            self.rag = RagTokenForGeneration(config)
        self.tokenizer = self.rag.generator.tokenizer
        self.max_new_tokens = config.evaluator.max_new_tokens
        self.prompt_template = config.evaluator.prompt_template
        self.dataset = get_prompt_dataset()

    def evaluate(self):
        for i, query in enumerate(tqdm(self.dataset)):
            if i == 2: break
            query_ids = self.tokenizer.encode(query)
            output_ids, _, passage0 = self.rag.generate(
                query_ids, max_new_tokens=self.max_new_tokens, template=self.prompt_template)
        
        run_output_dir = Path(self.config.output_dir, self.run_id)
        run_output_dir.mkdir(parents=True, exist_ok=True)
        stats_file = run_output_dir / f"stats.json"
        self.rag.stats.dump(stats_file)
        self.logger.info(f"Stats saved to `{stats_file}`")
        
if __name__ == "__main__":
    config = LatencyConfig()    
    config.parse_sys_args()
    config.generator.s_sequence = 1024
    config.retriever.s_context = 256
    config.retriever.n_docs = 1
    config.retriever.s_aggregate = 1
    config.retriever.downsample_type = 1
    config.sampler.do_sample = False
    
    evaluator = LatencyEvaluator(config)
    evaluator.evaluate()
