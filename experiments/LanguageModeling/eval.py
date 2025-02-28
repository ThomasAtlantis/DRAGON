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
    --evaluator.output_dir "outputs/" \
    --evaluator.dataset "Salesforce/wikitext,wikitext-2-raw-v1" \
    --evaluator.s_prefix 128 \
    --cache.load_index
"""

from tqdm import tqdm
from dragon.config import DragonConfig
from dragon.rag import RagSequenceForGeneration
from dragon.utils.data_process.data_utils import ContextualLMSampleLoader
from experiments.evaluator import Evaluator
from experiments.metrics import CrossEntropy
from experiments.utils import load_dataset

class LanguageModelingEvaluator(Evaluator):
    
    def __init__(self, config: DragonConfig):
        super().__init__(config, name="LanguageModeling")
        self.rag = RagSequenceForGeneration(config)
        repo_id, dataset_id = config.evaluator.dataset.split(",")
        self.data = load_dataset(
            repo_id, dataset_id, cache_path=config.cache.directory,
            split=f"test[0%:{int(config.evaluator.data_ratio * 100)}%]"
        )["text"]
        self.tokenizer = self.rag.generator.tokenizer
        self.metric = CrossEntropy(device=config.device)
    
    def data_loader(self):
        for i in tqdm(range(0, len(self.data), self.config.s_block)):
            texts = "\n\n".join(self.data[i: i + self.config.s_block])
            loader, total = ContextualLMSampleLoader(
                token_list=self.tokenizer.encode_plus(texts)["input_ids"],
                bos_token=self.rag.generator.context_switching_id,
                max_seq_len=self.rag.generator.max_seq_len,
                prefix_len=self.config.s_prefix)
            yield from tqdm(loader(), total=total)

    def evaluate(self):
        self.metric.reset()
        for query_ids, input_ids, label_ids in self.data_loader():
            logprobs, _ = self.rag(query_ids, input_ids)
            self.metric.update(logprobs=logprobs, labels=label_ids)
        result = float(self.metric.compute())
        self.logger.info(f"{self.metric.__class__.__name__}: {result:.4f}")
        self.save_output({"cross_entropy": result})
