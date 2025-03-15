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
import argparse
from tqdm import tqdm
from flashrag.utils import get_dataset
from flashrag.evaluator import Evaluator as FlashragEvaluator

from dragon.config import DragonConfig
from dragon.dragon import Dragon
from dragon.utils.stable import seed_everything
from dragon.utils.configure import Field as F
seed_everything(42)


def normalize(text):
    text = text.replace("_", "")
    text = text.replace(".", "")
    text = text.strip()
    return text

class QAConfig(DragonConfig):
    class evaluator:
        output_dir = F(str, default="outputs/", help="Output directory")

class QAEvaluator:

    def __init__(self, config: QAConfig):
        self.device = Dragon(config)
        self.prompt_template = "context: {context} given the context, answer the question wthin 5 words: {query}? The answer is: " 
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
            "save_dir": ".",
        }

        self.dataset = get_dataset(self.flashrag_config)
        self.dataset = self.dataset[self.flashrag_config["split"][0]]
        self.evaluator = FlashragEvaluator(self.flashrag_config)

    def evaluate(self):
        pred_answer_list = []
        while not self.device.ready_for_generation:
            time.sleep(0.1)
        for item in tqdm(self.dataset):
            query = item.question
            response = self.device.query(query, self.prompt_template, self.max_new_tokens)
            response = normalize(response)
            pred_answer_list.append(response)
        
        self.dataset.update_output("pred", pred_answer_list)
        evaluation_results = self.evaluator.evaluate(self.dataset)
        self.device.shutdown()
        print(evaluation_results)

if __name__ == "__main__":
    config = QAConfig()
    config.retriever.passages = "wikipedia[remote]"
    config.generator.model = "facebook/opt-1.3b"
    config.generator.s_sequence = 896
    config.retriever.s_context = 128
    config.retriever.n_docs = 4
    config.retriever.s_aggregate = 4
    config.sampler.do_sample = False
    config.trans.tx_port = 6000
    config.trans.rx_port = 5000
    
    args = argparse.ArgumentParser()
    args.add_argument('--rank', type=int, default=0, help='0 for cloud, 1 for device')
    args = args.parse_args()
    
    config.trans.rank = args.rank
    if args.rank == 0:
        config.trans.tx_port, config.trans.rx_port = config.trans.rx_port, config.trans.tx_port
        cloud = Dragon(config)
    else:
        evaluator = QAEvaluator(config)
        evaluator.evaluate()

