import csv
import json
import datasets
from tqdm import tqdm
from pathlib import Path
from typing import List, NamedTuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from datasets import Dataset
import re

from ..cache import file_cache
from ..mlogging import Logger

logger = Logger.build(__name__, level="INFO")


def chunkify(dataset, n):
    passages = []
    for line in tqdm(dataset, desc="Chunkifying", leave=False):
        text = line["text"].split()
        if len(text) < 10: continue
        for i in range(0, len(text), n):
            passages.append({
                "text": " ".join(text[i: i + n]),
                "id": len(passages)
            })
    return passages

def parse_wikitext(dataset: Dataset, depth=2):
    title_stack = []
    dataset_size = len(dataset)
    current_chunk = {"text": [], "start": 0, "end": 0}
    title_pattern = re.compile(r'^([=\s]+)(.*?)([\s=]+)$')
    end_block = False
    for line_num, line in enumerate(tqdm(dataset, desc="Parsing WikiText")):
        line = line['text'].strip()
        if title_match := title_pattern.match(line):
            level = len(title_match.group(1)) // 2 - 1
            if level > depth:
                current_chunk["text"].append(line)
            else:    
                title = title_match.group(2).strip()
                title_stack = title_stack[:level]
                title_stack.append(title)
                end_block = True
        elif line:
            current_chunk["text"].append(line)
        
        if (end_block or line_num == dataset_size - 1) and current_chunk["text"]:
            current_chunk["end"] = line_num - 1
            current_chunk["title"] = ', '.join(title_stack)
            yield current_chunk
            current_chunk = {"text": [], "start": line_num + 1}
            end_block = False

def chunkify_v2(dataset, chunk_size):  # Poor performance, never use this!
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, length_function=lambda x: len(x.split()),
        chunk_overlap=0, separators=[" ", ",", ".", "!", "?", ";"]
    )
    docs = []
    for item in parse_wikitext(dataset):
        content = '\n'.join(item["text"])
        if len(content.split()) < chunk_size:
            continue
        docs.append(Document(
            page_content=content, 
            title=item["title"]
        ))
    docs = text_splitter.split_documents(docs)
    docs = [doc for doc in docs if len(doc.page_content.split()) >= chunk_size * 0.8]

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
        passages = chunkify_v2(dataset, chunk_size)
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
