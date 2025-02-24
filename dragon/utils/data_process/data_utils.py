import csv
import datasets
from tqdm import tqdm
from pathlib import Path
import json
from typing import List, NamedTuple
Token = int

from ..cache import file_cache
from ..mlogging import Logger

logger = Logger.build(__name__, level="INFO")

def chunkify(dataset, n):
    passages = []
    for line in tqdm(dataset):
        text = line["text"].split()
        if len(text) < 10: continue
        for i in range(0, len(text), n):
            passages.append({
                "text": " ".join(text[i: i + n]),
                "id": len(passages) + 1
            })
    return passages


def load_passages_hf(repo_id, dataset_id, chunk_size, cache_path=".cache"):
    logger.info(f'Loading `{dataset_id}` as passages from Hugging Face Repo `{repo_id}` ...')
    @file_cache(Path(cache_path, dataset_id, "passages.txt"))
    def wrapper():
        dataset = datasets.load_dataset(repo_id, dataset_id, split='train')
        passages = chunkify(dataset, chunk_size)
        return passages
    return wrapper()


def load_passages_local(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")
    logger.info(f'Loading passages from local file `{path}` ...')
    with open(path) as ifstream:
        if path.endswith('.tsv'):
            reader = csv.DictReader(ifstream, delimiter='\t')
            passages = [row for row in tqdm(reader)]
        elif path.endswith('.jsonl'):
            passages = [json.loads(line) for line in tqdm(ifstream)]
        else:
            raise ValueError(f"Unsupported file format: {Path(path).suffix}")
    return passages


def load_passages(passages, chunk_size, cache_path=".cache"):
    if "," in passages:
        repo_id, dataset_id = passages.split(",")
        passages = load_passages_hf(repo_id, dataset_id, chunk_size, cache_path)
    else:
        passages = load_passages_local(passages)
    return passages


class DataSample(NamedTuple):
    """
    Prompt is the user query, while inputs and labels are for language modeling.
    """
    prompt: List[Token]
    inputs: List[Token]
    labels: List[Token]

def tokens_to_contextual_lm_samples_total(token_list, max_seq_len, context_len):
    total = len(token_list) - max_seq_len
    pred_len = max_seq_len - context_len + 1
    total = (total + pred_len - 1) // pred_len + 1
    return total


def tokens_to_contextual_lm_samples(
        token_list: List[Token], 
        bos_token: Token, 
        max_seq_len: int, 
        context_len: int
    ):
    """
    Yield data samples for contextual language modeling through rolling windows on the provided token sequence.

    @param token_list: List of tokens, usually some tokenized text
    @param bos_token: Beginning of sequence token as the initial context
    @param max_seq_len: Maximum sequence length of the language model
    @param context_len: Sequence length for the context

    @return: DataSample(prompt, inputs, labels)

    e.g.
    >>> for context, inputs, labels in tokens_to_contextual_lm_samples([1, 2, 3, 4, 5, 6, 7, 8, 9], 0, 4, 2):
    ...     print(f"Context: {context}, Inputs: {inputs}, Labels: {labels}")
    Context: [0], Inputs: [1, 2, 3], Labels: [1, 2, 3, 4]
    Context: [2, 3, 4], Inputs: [5], Labels: [5, 6]
    Context: [4, 5, 6], Inputs: [7], Labels: [7, 8]
    Context: [5, 6, 7, 8], Inputs: [], Labels: [9]
    """
    pred_len = max_seq_len - context_len + 1

    def extract_context(beg, end):
        end = min(end, len(token_list))
        return DataSample(
            prompt=token_list[end - max_seq_len - 1: beg - 1],
            inputs=token_list[beg - 1: end - 1],
            labels=token_list[beg: end]
        )

    yield DataSample(
        prompt=[], 
        inputs=[bos_token]+token_list[: max_seq_len - 1], 
        labels=token_list[: max_seq_len]
    )  # Initially, model the entire sequence given the bos_token
    for i in range(max_seq_len, len(token_list), pred_len):
        yield extract_context(i, i + pred_len)

def ContextualLMSampleLoader(
    token_list: List[Token], 
    bos_token: Token, 
    max_seq_len: int, 
    context_len: int):
    from functools import partial
    total = tokens_to_contextual_lm_samples_total(
        token_list, max_seq_len, context_len)
    loader = partial(tokens_to_contextual_lm_samples, 
        token_list, bos_token, max_seq_len, context_len)
    return loader, total
