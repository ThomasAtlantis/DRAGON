import pickle
import torch
import json
from pathlib import Path
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedModel
from dragon.utils.data_process import text_utils
from dragon.utils.data_process import data_utils
from dragon.retriever.models import contriever
from dragon.utils.configure import Configure
from dragon.utils.configure import Field as F
from dragon.utils.mlogging import Logger
from dragon.utils.data_process.text_utils import normalize


logger = Logger.build(__name__, level="INFO")

class GenEmbeddingConfig(Configure):
    """
    We use this script to generate embeddings for passages in a corpus. The optional arguments shard_id and num_shards
    are used to split the corpus into multiple shards and generate embeddings for each shard separately. This is useful
    when the corpus is too large to fit in memory. The script will generate embeddings for the shard with id shard_id
    out of num_shards shards. The embeddings are saved in the output_dir directory with the prefix prefix.
    """
    passages       = F(str, required=True, help="Passage file with suffix in ['.tsv', '.jsonl'] or"
                                                "Hugging Face RepoID and DatasetID, split with comma")
    retriever      = F(str, required=True, help="Repository or directory containing model weights and config file")

    output_dir     = F(str, default="wikipedia_embeddings", help="Directory to save embeddings")
    prefix         = F(str, default="passages", help="Prefix of embedding file name")
    shard_id       = F(int, default=0, help="Id of the current shard")
    num_shards     = F(int, default=1, help="Total number of shards")
    batch_size     = F(int, default=512, help="Batch size for encoding")
    passage_size   = F(int, default=512, help="Number of tokens in a passage sequence")
    chunk_size     = F(int, default=64, help="Number of words in a raw passage chunk")
    fp16           = F(bool, default=True, help="Inference in fp16")
    title          = F(bool, default=False, help="Add title to the passage body")
    lowercase      = F(bool, default=False, help="Lowercase text before encoding")
    normalize      = F(bool, default=False, help="Normalize text before encoding")

    class cache:
        directory  = F(str, default=".cache", help="Directory to save cache files")
     
config = GenEmbeddingConfig()
config.parse_sys_args()
logger.info(json.dumps(config.as_dict(), indent=2))


def concat_title_body(passage):
    text = passage['text']
    if config.title and 'title' in passage:
        text = passage['title'] + ' ' + text
    return text

def passage_loader(passages: list[str], batch_size: int, config: GenEmbeddingConfig):
    for k in range(0, len(passages), batch_size):
        batch_ids, batch_text = [], []
        for p in passages[k: k + batch_size]:
            batch_ids.append(p['id'])
            text = concat_title_body(p)            
            if config.lowercase:
                text = text.lower()
            if config.normalize:
                text = normalize(text)
            batch_text.append(text)
        yield batch_ids, batch_text


# def embed_passages(passages: list, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, config: GenEmbeddingConfig):
#     logger.info(f'Embedding {len(passages)} passages ...')
#     allids, allembeddings = [], []
#     total = (len(passages) + config.batch_size - 1) // config.batch_size
#     with torch.inference_mode():
#         for batch_ids, batch_text in tqdm(passage_loader(passages, config), total=total):
#             encoded_batch = tokenizer.batch_encode_plus(  # TODO: extract a unique definition
#                 batch_text, return_tensors="pt",
#                 max_length=config.passage_size,
#                 padding=True, truncation=True
#             )
#             encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
#             embeddings = model(**encoded_batch)

#             allids.extend(batch_ids)
#             allembeddings.append(embeddings)

#     allembeddings = torch.cat(allembeddings, dim=0).cpu().numpy()
#     logger.debug(f"allembeddings.shape={allembeddings.shape}")
#     logger.debug(f"allembeddings.dtype={allembeddings.dtype}")
#     return allids, allembeddings

def get_shard(passages):
    shard_size = len(passages) // config.num_shards
    beg_idx = config.shard_id * shard_size
    end_idx = min(beg_idx + shard_size, len(passages))
    logger.info(f'Getting shard {config.shard_id} `passages[{beg_idx}: {end_idx}]` ...')
    return passages[beg_idx: end_idx]


def main():
    logger.info(f'Loading retriever from `{config.retriever}`')
    model, tokenizer = contriever.load_retriever(config.retriever)
    model = model.cuda()

    if config.fp16:
        model = model.half()

    passages = get_shard(data_utils.load_passages(config.passages, config.chunk_size, cache_path=config.cache.directory))
    # allids, allembeddings = embed_passages(passages, model, tokenizer)
    allids, allembeddings = text_utils.embed_texts(  # TODO: move this function into Retriever
        model, tokenizer, passages, config.batch_size, config.passage_size, passage_loader, config=config)

    save_file = Path(config.output_dir) / f'{config.prefix}_{config.shard_id:02d}.pkl'
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f'Saving {len(allids)} passage embeddings to `{save_file}` ...')
    with open(save_file, mode='wb') as f:
        pickle.dump((allids, allembeddings), f)
    logger.info(f'{len(allids)} passages processed. Written to `{save_file}`.')


if __name__ == '__main__':
    """
    Usage Example:
    python -m dragon.toolbox.generate_passage_embeddings \
        --retriever "facebook/contriever" \
        --passages "Salesforce/wikitext,wikitext-2-raw-v1" \
        --output_dir data/embeddings \
        --batch_size 512 \
        --passage_size 128 \
        --chunk_size 64
    """
    main()
