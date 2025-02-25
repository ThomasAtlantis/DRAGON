import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

from dragon.config import DragonConfig
from dragon.decoder import Decoder
from dragon.retriever.retriever import Retriever
from dragon.generator.generator import Generator
from dragon.utils.mlogging import Logger

logger = Logger.build(__name__, level="INFO")
config = DragonConfig()
config.parse_sys_args()
logger.info(json.dumps(config.as_dict(), indent=2))


def compute_bpb(decoder: Decoder):
    all_logprobs = []
    data = load_dataset(
        config.repo_id, config.dataset_id, 
        split=f"test[0%:{int(config.data_ratio * 100)}%]"
    )
    data = data["text"]
    for i in tqdm(range(0, len(data), config.block_size)):
        batch = data[i: i + config.block_size]
        doc = "\n\n".join(batch)
        output = decoder.get_perplexity_data(doc)
        all_logprobs.append(output.logprobs)
    all_logprobs = np.concatenate(all_logprobs)
    logger.info(f"Computed logprobs: {all_logprobs.shape}")
    bpb = -np.mean(all_logprobs)
    return bpb

def main():
    generator = Generator(config)
    retriever = Retriever(config)
    decoder = Decoder(generator=generator, retriever=retriever, config=config)

    Path(config.output_path).parent.mkdir(parents=True, exist_ok=True)
    bpb = compute_bpb(decoder=decoder)
    logger.info(f"Bits per Byte: {bpb:.4f}")


if __name__ == "__main__":
    """
    Usage Example:
    python test.py \
        --repo_id "Salesforce/wikitext" \
        --passages "Salesforce/wikitext,wikitext-103-raw-v1" \
        --dataset_id "wikitext-103-raw-v1" \
        --model_config_path facebook/opt-125m  \
        --passages_embeddings "./data/embeddings/" \
        --re_model_name_or_path "facebook/contriever" \
        --retrieved_max_length 128 \
        --context_len 128 \
        --pred_len 768 \
        --output_path outputs/ppl.data \
        --ensemble 10 \
        --n_docs 10
    """
    main()