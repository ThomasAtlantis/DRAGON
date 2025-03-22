import sys, os
sys.path.append(".")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from pathlib import Path
import argparse
from tqdm import tqdm
from flashrag.utils import get_dataset
from flashrag.dataset import Dataset
from flashrag.evaluator import Evaluator as FlashragEvaluator

from dragon.config import DragonConfig
from dragon.utils.stable import seed_everything
from dragon.utils.configure import Field as F
from dragon.baselines.centralized_rag import RagSequenceForGeneration, RagTokenForGeneration

seed_everything(42)


def normalize(text: str, question: str):
    text = text.lower()
    text = text.split("\n")[0]
    text = text.split("<")[0]
    return text

class QAConfig(DragonConfig):
    class evaluator:
        output_dir = F(str, default="outputs/", help="Output directory")
        method = F(str, default="seq", help="Method for evaluation: seq or tok")

class QAEvaluator:
    
    def __init__(self, config: QAConfig):
        if config.evaluator.method == "seq":
            self.rag = RagSequenceForGeneration(config)
        else:
            self.rag = RagTokenForGeneration(config)
        self.tokenizer = self.rag.generator.tokenizer
        self.prompt_template = "context: {context}.\ngiven the context, answer the question within 5 words: {query}?\n"
        self.max_new_tokens = 10

        self.flashrag_config = {
            "dataset_path": "datasets/nq",
            "split": ["dev"],
            "test_sample_num": 100,
            "random_sample": False,
            "dataset_name": "nq",
            "save_metric_score": True,
            "save_intermediate_data": True,
            "metrics": ["f1", "EM"],
            "save_dir": f"outputs/CRAG-{config.evaluator.method}",
        }
        self.flashrag_config["save_dir"] = "outputs/tmp/"
        Path(self.flashrag_config["save_dir"]).mkdir(parents=True, exist_ok=True)
        # split_path = Path(self.flashrag_config["dataset_path"]) / f"{self.flashrag_config['split'][0]}.jsonl"
        # self.dataset = Dataset(
        #     self.flashrag_config, str(split_path), 
        #     sample_num=self.flashrag_config["test_sample_num"], 
        #     random_sample=self.flashrag_config["random_sample"]
        # )
        self.dataset = get_dataset(self.flashrag_config)
        self.dataset = self.dataset[self.flashrag_config["split"][0]]
        self.evaluator = FlashragEvaluator(self.flashrag_config)

    def evaluate(self):
        pred_answer_list = []
        passage_list = []
        for item in tqdm(self.dataset):
            query = item.question
            query_ids = self.tokenizer.encode(query)
            
            output_ids, _, passage0 = self.rag.generate(query_ids, max_new_tokens=self.max_new_tokens, template=self.prompt_template)
            output_txt = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            passage_list.append(passage0)
            output_txt = normalize(output_txt, question=query)
            pred_answer_list.append(output_txt)
        
        self.dataset.update_output("pred", pred_answer_list)
        self.dataset.update_output("passage", passage_list)
        evaluation_results = self.evaluator.evaluate(self.dataset)
        print(evaluation_results)
        
        
if __name__ == "__main__":
    config = QAConfig()
    config.retriever.passages = "wikipedia[remote]"
    # config.generator.model = "facebook/opt-1.3b"
    # config.generator.model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    config.generator.model = "microsoft/Phi-4-mini-instruct"
    config.generator.s_sequence = 896
    config.retriever.s_context = 256
    config.retriever.n_docs = 4
    config.retriever.s_aggregate = 4
    config.sampler.do_sample = False
    
    args = argparse.ArgumentParser()
    args.add_argument('--method', type=str, default="seq", help='Method for evaluation: seq or tok')
    args = args.parse_args()

    config.evaluator.method = args.method
    evaluator = QAEvaluator(config)
    evaluator.evaluate()
