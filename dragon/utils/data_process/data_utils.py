import csv
import json
import re
import datasets
from datasets import Dataset
from tqdm import tqdm
from pathlib import Path
from typing import List, NamedTuple, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from ..cache import file_cache
from ..mlogging import Logger

logger = Logger.build(__name__, level="INFO")


def parse_wikitext(dataset: Dataset) -> List[Dict]:
    title_stack, chunks = [], []
    dataset_size = len(dataset)
    current_chunk = {"text": [], "start": 0, "end": 0}
    title_pattern = re.compile(r'^([=\s]+)(.*?)([\s=]+)$')
    for line_num, line in enumerate(dataset):
        line = line['text'].strip()
        if title_match := title_pattern.match(line):
            level = len(title_match.group(1)) // 2 - 1
            title = title_match.group(2).strip()
            title_stack = title_stack[:level]
            title_stack.append(title)
        elif line:
            current_chunk["text"].append(line)
        
        if (title_match or line_num == dataset_size - 1) and current_chunk["text"]:
            current_chunk["end"] = line_num - 1
            current_chunk["title"] = ', '.join(title_stack)
            chunks.append(current_chunk)
            current_chunk = {"text": [], "start": line_num + 1}
    return chunks

def chunkify(dataset, n):
    passages = []
    for line in tqdm(dataset):
        text = line["text"].split()
        if len(text) < 10: continue
        for i in range(0, len(text), n):
            passages.append({
                "text": " ".join(text[i: i + n]),
                "id": len(passages)
            })
    return passages

def chunkify_v2(dataset, chunk_size):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, length_function=lambda x: len(x.split()),
        separators=["\n", ".", "!", "?", ";", ",", " "]
    )
    docs = [
        Document(
            page_content='\n'.join(item["text"]), 
            title=item["title"]
        ) for item in parse_wikitext(dataset)
    ]
    docs = text_splitter.split_documents(docs)

    # from langchain_community.document_transformers import EmbeddingsRedundantFilter
    # from langchain_huggingface import HuggingFaceEmbeddings
    # redundant_filter = EmbeddingsRedundantFilter(
    #     embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
    # docs = redundant_filter.transform_documents(docs)

    passages = [
        { "text": doc.page_content, "id": i } for i, doc in enumerate(docs)
    ]
    logger.info(f"Chunkified {len(docs)} documents into {len(passages)} passages.")
    return passages


def load_passages_hf(repo_id, dataset_id, chunk_size, cache_path=".cache"):
    logger.info(f'Loading `{dataset_id}` as passages from Hugging Face Repo `{repo_id}` ...')
    @file_cache(Path(cache_path, dataset_id, "passages.jsonl"))
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
    query: List[int]
    input: List[int]
    label: List[int]


def tokens_to_contextual_lm_samples_total(token_list, max_seq_len, prefix_len):
    total = len(token_list) - max_seq_len
    pred_len = max_seq_len - prefix_len + 1
    total = (total + pred_len - 1) // pred_len + 1
    return total


def tokens_to_contextual_lm_samples(
        token_list: List[int], 
        bos_token: int, 
        max_seq_len: int, 
        prefix_len: int
    ):
    """
    Yield data samples for contextual language modeling through rolling windows on the provided token sequence.

    @param token_list: List of tokens, usually some tokenized text
    @param bos_token: Beginning of sequence token as the initial query
    @param max_seq_len: Maximum sequence length of the language model
    @param prefix_len: Sequence length for the prefix (query)

    @return: DataSample(query, input, label)

    e.g.
    >>> for query, input, label in tokens_to_contextual_lm_samples([1, 2, 3, 4, 5, 6, 7, 8, 9], 0, 4, 2):
    ...     print(f"query: {query}, input: {input}, label: {label}")
    query: [0], input: [1, 2, 3], label: [1, 2, 3, 4]
    query: [2, 3, 4], input: [5], label: [5, 6]
    query: [4, 5, 6], input: [7], label: [7, 8]
    query: [5, 6, 7, 8], input: [], label: [9]
    """
    pred_len = max_seq_len - prefix_len + 1

    def extract_context(beg, end):
        end = min(end, len(token_list))
        return DataSample(
            query=token_list[end - max_seq_len - 1: beg - 1],
            input=token_list[beg - 1: end - 1],
            label=token_list[beg: end]
        )

    yield DataSample(
        query=[], 
        input=[bos_token]+token_list[: max_seq_len - 1], 
        label=token_list[: max_seq_len]
    )  # Initially, model the entire sequence given the bos_token
    for i in range(max_seq_len, len(token_list), pred_len):
        yield extract_context(i, i + pred_len)

def ContextualLMSampleLoader(
    token_list: List[int], 
    bos_token: int, 
    max_seq_len: int, 
    prefix_len: int):
    from functools import partial
    total = tokens_to_contextual_lm_samples_total(
        token_list, max_seq_len, prefix_len)
    loader = partial(tokens_to_contextual_lm_samples, 
        token_list, bos_token, max_seq_len, prefix_len)
    return loader, total
