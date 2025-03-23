import sys; sys.path.append(".")
from dragon.utils.stable import seed_everything
from dragon.utils.mlogging import Logger


seed_everything(42)
logger = Logger.build(__name__, level="INFO")


def in_adjacent_area(predicted, ground_truth, epsilon=0.05):
    l_bound, r_bound = ground_truth - epsilon, ground_truth + epsilon
    return l_bound <= predicted <= r_bound


def generate_embeddings():
    import subprocess
    script = """
    HF_ENDPOINT=https://hf-mirror.com \
    python -m dragon.toolbox.embed_passages \
        --retriever.model "contriever" \
        --retriever.passages "Salesforce/wikitext,wikitext-2-raw-v1" \
        --output_dir "./data/wikitext2" \
        --retriever.bs_encode 512 \
        --retriever.s_passage 64 \
        --text.with_title
    """
    subprocess.run(script, shell=True)


def clear_cache():
    from pathlib import Path
    cache_file = Path(".cache/wikitext-2-raw-v1/passages.jsonl")
    if cache_file.exists(): cache_file.unlink()


def test_ppl():
    import os
    from experiments.LanguageModeling.eval import (
        LanguageModelingEvaluator as Evaluator, 
        LanguageModelingConfig as Config
    )

    clear_cache()
    generate_embeddings()

    # run evaluation
    config = Config()
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    config.retriever.passages = "Salesforce/wikitext,wikitext-2-raw-v1"
    config.retriever.passages_embeddings = "data/wikitext2/*.pkl"
    config.generator.model = "facebook/opt-1.3b"
    config.generator.s_sequence = 896
    config.evaluator.output_dir = "outputs/"
    config.evaluator.dataset = "Salesforce/wikitext,wikitext-2-raw-v1"
    config.evaluator.s_prefix = 128
    config.retriever.s_context = 128
    config.retriever.n_docs = 16
    config.retriever.s_aggregate = 16

    class TestEvaluator(Evaluator):
        def evaluate(self):
            self.metric.reset()
            for i, (query_ids, input_ids, label_ids) in enumerate(self.data_loader()):
                if 10 <= i < 20:
                    logprobs = self.rag(query_ids, input_ids, template=self.template)
                    self.metric.update(logprobs=logprobs, labels=label_ids)
                if i == 20: break
            return float(self.metric.compute())
    
    result = TestEvaluator(config).evaluate()
    logger.info(result)
    assert in_adjacent_area(result, 2.5228, epsilon=0.01)
