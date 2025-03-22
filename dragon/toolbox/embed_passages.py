import pickle
from pathlib import Path
from dragon.retriever import CustomRetriever as Retriever
from dragon.utils.data_process import text_utils
from dragon.utils.data_process import data_utils
from dragon.utils.mlogging import Logger
from dragon.config import DragonConfig
from dragon.utils.configure import Configure
from dragon.utils.configure import Field as F

class EmbedPassagesConfig(Configure):
    """
    We use this script to generate embeddings for passages in a corpus. The optional arguments shard_id and num_shards
    are used to split the corpus into multiple shards and generate embeddings for each shard separately. This is useful
    when the corpus is too large to fit in memory. The script will generate embeddings for the shard with id shard_id
    out of num_shards shards. The embeddings are saved in the output_dir directory with the prefix prefix.
    """
    retriever         = DragonConfig.retriever
    text              = DragonConfig.text
    fp16              = DragonConfig.fp16
    device            = DragonConfig.device
    
    output_dir        = F(str, default="wikipedia_embeddings", help="Directory to save embeddings")
    prefix            = F(str, default="passages", help="Prefix of embedding file name")
    shard_id          = F(int, default=0, help="Id of the current shard")
    num_shards        = F(int, default=1, help="Total number of shards")
    cache             = DragonConfig.cache

logger = Logger.build(__name__, level="INFO")
config = EmbedPassagesConfig()
config.parse_sys_args()
# logger.info(json.dumps(config.as_dict(), indent=2))

def passage_processor(passages):
    for i, passage in enumerate(passages):
        text: str = passage['text']
        if config.text.with_title and 'title' in passage:
            text = passage['title'] + ' ' + text
        if config.text.lowercase:
            text = text.lower()
        if config.text.normalize:
            text = text_utils.normalize(text)
        passages[i] = text
    return passages


def get_shard(passages):
    shard_size = len(passages) // config.num_shards
    beg_idx = config.shard_id * shard_size
    end_idx = min(beg_idx + shard_size, len(passages))
    logger.info(f'Getting shard {config.shard_id} `passages[{beg_idx}: {end_idx}]` ...')
    return passages[beg_idx: end_idx]


def main():
    retriever = Retriever(config)
    passages = get_shard(data_utils.load_passages(
        config.retriever.passages, config.retriever.s_passage, 
        cache_path=config.cache.directory))
    allids = [passage['id'] for passage in passages]
    allembeddings = retriever.embed_texts(
        passages, batch_size=config.retriever.bs_encode, 
        post_processor=passage_processor, progress=True
    )

    save_file = Path(config.output_dir) / f'{config.prefix}_{config.shard_id:02d}.pkl'
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f'Saving {len(allids)} passage embeddings to `{save_file}` ...')
    with open(save_file, mode='wb') as f:
        pickle.dump((allids, allembeddings), f)
    logger.info(f'{len(allids)} passages processed. Written to `{save_file}`.')


if __name__ == '__main__':
    main()
