"""
HF_ENDPOINT=https://hf-mirror.com \
python -u run.py \
    --retriever.passages "Salesforce/wikitext,wikitext-2-raw-v1" \
    --retriever.passages_embeddings "data/wikitext2/*.pkl" \
    --retriever.s_context 128 \
    --retriever.n_docs 2 \
    --retriever.s_aggregate 2 \
    --generator.model "facebook/opt-1.3b"  \
    --generator.s_sequence 896 \
    --cache.load_index \
    --evaluator.max_new_tokens 15
"""

from dragon.config import DragonConfig
from dragon.utils.configure import Field as F
from dragon.rag import RagSequenceForGeneration, RagTokenForGeneration
from experiments.evaluator import Evaluator

class SeqAggVsTokAggConfig(DragonConfig):
    class evaluator:
        max_new_tokens = F(int, default=10, help="Maximum number of tokens to generate")
        output_dir = F(str, default="outputs/", help="Output directory")

class SeqAggVsTokAggEvaluator(Evaluator):
    
    def __init__(self, config: SeqAggVsTokAggConfig):
        super().__init__(config, name="SeqAggVsTokAgg")
        self.rag_seq = RagSequenceForGeneration(config)
        self.rag_tok = RagTokenForGeneration(config)
        self.tokenizer = self.rag_seq.generator.tokenizer
        self.max_new_tokens = config.evaluator.max_new_tokens

    def evaluate(self):
        results = []
        queries = [
            "I love China, because",
            "when did us get involved in vietnam war"
        ]
        for query in queries:
            query_ids = self.tokenizer.encode(query)

            output_ids_seq, _ = self.rag_seq.generate(query_ids, [], max_new_tokens=self.max_new_tokens)
            output_seq = self.tokenizer.decode(output_ids_seq, skip_special_tokens=True)
            
            output_ids_tok, _ = self.rag_tok.generate(query_ids, [], max_new_tokens=self.max_new_tokens)
            output_tok = self.tokenizer.decode(output_ids_tok, skip_special_tokens=True)

            self.logger.info(f"Query: {query}")
            self.logger.info(f"Output (Seq): {output_seq}")
            self.logger.info(f"Output (Tok): {output_tok}")

            results.append({
                "query": query,
                "output_seq": output_seq,
                "output_tok": output_tok
            })

        self.save_output(results)
