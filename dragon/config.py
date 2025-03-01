from .utils.configure import Configure
from .utils.configure import Field as F

class DragonConfig(Configure):

    device                  = F(str,  default="cuda:0", help="Device to use for inference")
    random_seed             = F(int,  default=0,     help="random document")
    fp16                    = F(bool, default=True,  help="Inference in fp16")

    class retriever:
        model               = F(str,  default="contriever", help="The retriever class defined in dragon/retriever/models")
        bs_encode           = F(int,  default=512,   help="Batch size for encoding")
        s_passage           = F(int,  default=512,   help="Number of tokens in a passage sequence")
        s_passage_chunk     = F(int,  default=64,    help="Maximum number of words in a chunk")
        s_aggregate         = F(int,  default=0,     help="Number of documents to retrieve per questions")
        n_docs              = F(int,  default=10,    help="Number of documents to retrieve per questions")
        s_query             = F(int,  default=128,   help="Maximum number of tokens in a query")
        s_context           = F(int,  default=256,   help="Maximum number of tokens in a context")
        passages            = F(str,  required=True, help="Passage file with suffix in ['.tsv', '.jsonl'] or"
                                                         "Hugging Face RepoID and DatasetID, split with comma")
        passages_embeddings = F(str,  default="data/embeddings/*.pkl", help="Glob path to encoded passages")

    class indexer:
        s_embedding         = F(int,  default=768,   help="The embedding dimension for indexing")
        n_subquantizers     = F(int,  default=0,     help="Number of subquantizer used for vector quantization, if 0 flat index is used")
        n_bits              = F(int,  default=8,     help="Number of bits per subquantizer")
        bs_indexing         = F(int,  default=1e6,   help="Batch size of the number of passages indexed")
        
    class text:
        with_title          = F(bool, default=False, help="Add title to the passage body")
        lowercase           = F(bool, default=False, help="Lowercase text before encoding")
        remove_broken_sents = F(bool, default=False, help="If enabled, remove broken sentences")
        round_broken_sents  = F(bool, default=False, help="If enabled, round broken sentences")
        normalize           = F(bool, default=False, help="Normalize text")

    class generator:
        model               = F(str,  required=True, help="Path to the model configuration file")
        s_sequence          = F(int,  default=896,   help="")
    
    class sampler:
        do_sample           = F(bool, default=False, help="If enabled, use sampling instead of greedy decoding")
        temperature         = F(float,default=1.0,   help="Temperature for sampling")
        top_k               = F(int,  default=50,    help="Top-k sampling")
        top_p               = F(float,default=1.0,   help="Top-p sampling")

    class cache:
        directory           = F(str,  default=".cache", help="Directory to store cache files")
        load_query2docs     = F(bool, default=False, help="Load query2docs cache")
        dump_query2docs     = F(bool, default=False, help="Dump query2docs cache")
        load_index          = F(bool, default=False, help="If enabled, load index from disk")
        dump_index          = F(bool, default=False, help="If enabled, save index to disk")
