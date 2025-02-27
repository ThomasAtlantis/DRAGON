import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime
import sys

from dragon.config import DragonConfig
from dragon.rag import RagSequenceForGeneration
from dragon.utils.cache import file_cache
from dragon.utils.mlogging import Logger

logger = Logger.build(__name__, level="INFO")
config = DragonConfig()
config.parse_sys_args()
config_dict = config.as_dict()

def load_dataset(repo_id, dataset_id, split, cache_path=".cache"):
    @file_cache(Path(cache_path, dataset_id, "test_data.pkl"))
    def wrapper():
        import datasets
        dataset = datasets.load_dataset(repo_id, dataset_id, split=split)
        return dataset
    return wrapper()


def compute_bpb(rag: RAG):
    all_logprobs = []
    repo_id, dataset_id = config.evaluator.dataset.split(",")
    data = load_dataset(
        repo_id, dataset_id, 
        split=f"test[0%:{int(config.evaluator.data_ratio * 100)}%]",
        cache_path=config.cache.directory
    )["text"]
    for i in tqdm(range(0, len(data), config.evaluator.s_block)):
        doc = "\n\n".join(data[i: i + config.evaluator.s_block])
        logprobs = rag.get_perplexity_data(doc)
        all_logprobs.append(logprobs)
    all_logprobs = np.concatenate(all_logprobs)
    logger.info(f"Computed logprobs: {all_logprobs.shape}")
    bpb = -np.mean(all_logprobs)
    return float(bpb)

def save_output(output):
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    run_output_dir = Path(config.evaluator.output_dir, run_id)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    script = " ".join(["python", "-u"] + sys.argv)
    script = script.replace("--", "\\\n    --")

    with open(run_output_dir / "output.json", "w") as f:
        json.dump(output, f)
    with open(run_output_dir / "config.json", "w") as f:
        json.dump(config_dict, f)
    with open(run_output_dir / "run.sh", "w") as f:
        f.write(script)

    logger.info(f"Output saved to `{run_output_dir}`.")

def main():
    rag = RagSequenceForGeneration(config)
    bpb = compute_bpb(rag)
    logger.info(f"Bits per Byte: {bpb:.4f}")
    save_output({"bpb": bpb})

if __name__ == "__main__":
    main()