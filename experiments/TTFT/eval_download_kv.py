import sys, os
sys.path.append(".")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
from pathlib import Path
import queue
import time

from tqdm import trange
from dragon.queues import DraftItem

from dragon.config import DragonConfig
from dragon.utils.meter import Statistics, TimeMeter
from dragon.baselines.centralized_rag import RagForGeneration
from dragon.utils.stable import seed_everything
from dragon.utils.configure import Field as F
from dragon.transceiver import Message, Transceiver
from experiments.evaluator import Evaluator
seed_everything(42)

time_meter = TimeMeter()

def get_prompt_dataset(total=2):
    with open("prompts.json", "r") as f:
        prompts = json.load(f)[: total]
    return prompts

class TTFTConfig(DragonConfig):
    class evaluator:
        output_dir = F(str, default="outputs/", help="Output directory")
        max_new_tokens = F(int, default=100, help="Maximum number of tokens to generate")
        prompt_template = F(str, default="Context: {context}.\nInstruction: {query}\nAnswer: ", help="Prompt template")
        n_prompts = F(int, default=2, help="Number of prompts to evaluate")

class TTFTEvaluator(Evaluator):

    def __init__(self, config: TTFTConfig):
        super().__init__(config, name="TTFT")
        self.ready_for_generation = False
        self.prompt_template = config.evaluator.prompt_template
        self.max_new_tokens = config.evaluator.max_new_tokens
        self.dataset = get_prompt_dataset(config.evaluator.n_prompts)
        self.dragon_config = config
        self.rag = RagForGeneration(config)
        self.context_input_ids_list = []
        self.attention_mask_list = []
        self.scores_list = []
        for item in self.dataset:
            query = item["query"]
            query_ids = self.rag.generator.tokenizer.encode(query)
            context_input_ids, attention_mask, scores, _ = self.rag._prepare_inputs_for_generation(
                query_ids, [], self.prompt_template)
            self.context_input_ids_list.append(context_input_ids)
            self.attention_mask_list.append(attention_mask)
            self.scores_list.append(scores)

        self.stats = Statistics()
        self.draft_loc = queue.Queue(0)
        self.download_complete = queue.Queue(0)
        self.start_time = None
        self.transceiver = Transceiver(config)
        self.transceiver.register_observers(self._collect_observers())
        self.transceiver.send(Message.READY_FOR_GENERATION, None)

    def _collect_observers(self):
        return [
            self._rx_ready_for_generation,
            self._rx_kv_cache,
            self._rx_begin_generate,
            self._rx_shutdown,
        ]
    
    def _rx_ready_for_generation(self, mtype: int, mbody: object):
        if mtype != Message.READY_FOR_GENERATION: return False
        self.ready_for_generation = True
        self.logger.info("Remote is ready for generation.")
        return True
    
    def _send_kv_cache(self, past_key_values: DraftItem):
        self.logger.info("Sending key-values cache...")
        past_key_values = list(past_key_values)
        for i, _ in enumerate(past_key_values):
            past_key_values[i] = list(past_key_values[i])
            past_key_values[i][0] = past_key_values[i][0].cpu().float().numpy()
            past_key_values[i][1] = past_key_values[i][1].cpu().float().numpy()
            past_key_values[i] = tuple(past_key_values[i])
        past_key_values = tuple(past_key_values)
        self.transceiver.send(Message.KV_CACHE, past_key_values)
    
    def _rx_kv_cache(self, mtype: int, mbody: object):
        if mtype != Message.KV_CACHE: return False
        self.logger.info("Received key-values cache.")
        self.draft_loc.get()
        elapsed_time = time.time() - self.start_time
        self.download_complete.put(elapsed_time)
    
    def _rx_begin_generate(self, mtype: int, mbody: object):
        if mtype != Message.BEGIN_GENERATE: return False
        self.stats.new_record()
        query, prompt_template, i = mbody
        time.sleep(0.0526)  # simulate retrieval
        next_token, logprobs, past_key_value = self.rag._generate(
        self.context_input_ids_list[i], self.attention_mask_list[i], self.scores_list[i], n_logits=1)
        logprobs = logprobs[0]
        self._send_kv_cache(past_key_value)

    def _rx_shutdown(self, mtype: int, mbody: object):
        if mtype != Message.SHUTDOWN: return False
        self.logger.info("Received shutdown signal.")
        self.shutdown()
    
    def shutdown(self):
        run_output_dir = Path(self.config.output_dir, self.run_id)
        run_output_dir.mkdir(parents=True, exist_ok=True)
        stats_file = run_output_dir / f"stats.json"
        self.stats.dump(stats_file)
        self.logger.info(f"Stats saved to `{stats_file}`")

        self.logger.info("Shutting down...")
        self.transceiver.terminate()
    
    def evaluate(self):
        while not self.ready_for_generation:
            time.sleep(0.1)
        for i in trange(len(self.context_input_ids_list)):
            self.start_time = time.time()
            self.transceiver.send(Message.BEGIN_GENERATE, (
                self.dataset[i], self.prompt_template, i))
            self.logger.info("Start evaluation...")
            self.stats.new_record()
            time.sleep(0.0526)  # simulate retrieval
            self.draft_loc.put(None)
            elapsed_time = self.download_complete.get()
            self.stats.update(name="DownloadKV", stat=elapsed_time)
        
        if self.dragon_config.trans.rank == 1:
            self.transceiver.send(Message.SHUTDOWN, None)
            self.shutdown()

if __name__ == "__main__":
    config = TTFTConfig()
    config.parse_sys_args()
    config.generator.s_sequence = 1024
    config.retriever.s_context = 256
    config.sampler.do_sample = False
    config.trans.tx_port = 6000
    config.trans.rx_port = 6001

    if config.trans.rank == 0:
        config.trans.tx_host = "192.168.1.115"
        config.trans.tx_port, config.trans.rx_port = config.trans.rx_port, config.trans.tx_port
        evaluator = TTFTEvaluator(config)
    else:
        config.trans.tx_host = "192.168.1.126"
        evaluator = TTFTEvaluator(config)
        evaluator.evaluate()
