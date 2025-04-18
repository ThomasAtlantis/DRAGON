from .utils.configure import Configure
from .utils.configure import Field as F

class DragonConfig(Configure):

    device                  = F(str,  default="cuda:0", help="Device to use for inference")
    random_seed             = F(int,  default=0,     help="random document")
    fp16                    = F(bool, default=True,  help="Inference in fp16")

    class retriever:
        model               = F(str,  default="contriever", help="The retriever class defined in dragon/retriever/models")
        bs_encode           = F(int,  default=512,   help="Batch size for encoding")
        s_passage           = F(int,  default=64,    help="Number of words in a passage")
        s_aggregate         = F(int,  default=0,     help="Number of documents to retrieve per questions")
        n_docs              = F(int,  default=10,    help="Number of documents to retrieve per questions")
        s_context           = F(int,  default=256,   help="Maximum number of tokens in a context")
        passages            = F(str,  required=True, help="Passage file with suffix in ['.tsv', '.jsonl'] or"
                                                         "Hugging Face RepoID and DatasetID, split with comma")
        downsample_type     = F(int,  default=0,     help="Downsample type, 0 for no downsample, 1 for first half of topk, 2 for second half of topk")
        passages_embeddings = F(str,  default="data/embeddings/*.pkl", help="Glob path to encoded passages")
        host                = F(str,  default="192.168.1.126", help="Host address for the retriever")
        port                = F(int,  default=8765,  help="Port number for the retriever")

    class indexer:
        s_embedding         = F(int,  default=768,   help="The embedding dimension for indexing")
        n_subquantizers     = F(int,  default=0,     help="Number of subquantizer used for vector quantization, if 0 flat index is used")
        n_bits              = F(int,  default=8,     help="Number of bits per subquantizer")
        bs_indexing         = F(int,  default=100000,help="Batch size of the number of passages indexed")
        
    class text:
        with_title          = F(bool, default=False, help="Add title to the passage body")
        lowercase           = F(bool, default=False, help="Lowercase text before encoding")
        remove_broken_sents = F(bool, default=False, help="If enabled, remove broken sentences")
        round_broken_sents  = F(bool, default=False, help="If enabled, round broken sentences")
        normalize           = F(bool, default=False, help="Normalize text")

    class generator:
        model               = F(str,  required=True, help="Path to the model configuration file")
        s_sequence          = F(int,  default=896,   help="")
        use_fp16            = F(bool, default=False,  help="Use fp16 for generation")

    class reranker:
        do_rerank           = F(bool, default=False, help="If enabled, rerank the documents")
        model               = F(str,  default="cross-encoder/ms-marco-MiniLM-L6-v2", help="The reranker model name")
        period              = F(int,  default=0,     help="Number of steps between reranking")
        momentum            = F(float,default=0.0,   help="Weight to preserve the previous scores")
    
    class sampler:
        do_sample           = F(bool, default=False, help="If enabled, use sampling instead of greedy decoding")
        temperature         = F(float,default=1.0,   help="Temperature for sampling")
        top_k               = F(int,  default=50,    help="Top-k sampling")
        top_p               = F(float,default=1.0,   help="Top-p sampling")
    
    class aggregator:
        mode                = F(str,  default="synchronized", help="Aggregation mode")

    class profiler:
        n_epochs            = F(int, default=3, help="Number of epochs to evaluate the latency offline")
    
    class trans:
        rank = F(int, default=0, help="Index of the transceiver")
        tx_host = F(str, default="192.168.1.115", help="Remote host ip address")
        tx_port = F(int, default=5555, help="Port for sending data")
        rx_host = F(str, default="0.0.0.0", help="local host ip address")
        rx_port = F(int, default=5556, help="Port for receiving data")

    class cache:
        directory           = F(str,  default=".cache", help="Directory to store cache files")
        load_query2docs     = F(bool, default=False, help="Load query2docs cache")
        dump_query2docs     = F(bool, default=False, help="Dump query2docs cache")
        load_index          = F(bool, default=False, help="If enabled, load index from disk")
        dump_index          = F(bool, default=False, help="If enabled, save index to disk")
