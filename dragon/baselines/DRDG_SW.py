import queue
import time

import torch
from dragon.baselines.centralized_rag import RagForGeneration
from dragon.config import DragonConfig
from dragon.transceiver import Message, Transceiver
from dragon.utils.meter import Statistics, TimeMeter
from dragon.utils.mlogging import Logger


logging_level = "INFO"
time_meter = TimeMeter()

class RagSeqWise(RagForGeneration):
    def __init__(self, config: DragonConfig):
        super().__init__(config)
        self.logger = Logger.build(__class__.__name__, level=logging_level)
        self.input_queue = []
    
    def retrieval_and_prefilling(self, query: str, max_new_tokens: int, template: str):
        query_ids = self.generator.tokenizer.encode(query)
        context_input_ids, attention_mask, scores, doc_texts = self._prepare_inputs_for_generation(
            query_ids, [], template)
        last_token, logits, past_key_values = self.prefilling(context_input_ids, attention_mask)
        self.input_queue.append((
            last_token, logits, attention_mask, past_key_values, scores, max_new_tokens
        ))

    def prefilling(self, input_ids, attention_mask):
        output = self.generator(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits[0][-1]
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        last_token = self.generator.sampler(torch.exp(logprobs))
        return last_token, logits, output.past_key_values
    
    def decoding(self):
        last_token, logits, attention_mask, past_key_values, scores, max_new_tokens = self.input_queue.pop(0)
        logits_list = []
        for step in range(max_new_tokens - 1):
            self.logger.info(f"Decoding step={step} ...")
            input_ids = torch.as_tensor([last_token], dtype=torch.long, device=self.device)
            input_ids = input_ids.repeat(self.aggregate_size, 1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(input_ids)], dim=1)
            output = self.generator(
                input_ids=input_ids, attention_mask=attention_mask, 
                past_key_values=past_key_values
            )
            past_key_values = output.past_key_values
            logprobs = torch.nn.functional.log_softmax(output.logits.squeeze(), dim=-1)
            last_token = self.generator.sampler(torch.exp(logprobs))
            logits_list.append(output.logits.squeeze())
        logits_list.insert(0, logits)
        logits_list = torch.stack(logits_list)
        return logits_list, scores

    
class DRDG_SW:
    def __init__(self, config: DragonConfig):
        self.logger = Logger.build(__class__.__name__, level=logging_level)
        self.ready_for_generation = False
        self.rag = RagSeqWise(config)
        self.draft_loc = queue.Queue(0)
        self.draft_rem = queue.Queue(0)
        self.transceiver = Transceiver(config)
        self.transceiver.register_observers(self._collect_observers())
        self.transceiver.send(Message.READY_FOR_GENERATION, None)
        self.stats = Statistics()

    def shutdown(self):
        self.transceiver.send(Message.SHUTDOWN, None)
        time.sleep(3)
        self._shutdown()

    def _shutdown(self):
        self.logger.info("Shutting down.")
        self.transceiver.terminate()

    def query(self, query: str, template: str, max_new_tokens: int):
        self.stats.new_record()
        self.remote_prepare_complete = False
        self._send_begin_generate(query, template, max_new_tokens)
        self.rag.retrieval_and_prefilling(query, max_new_tokens, template)
        while not self.remote_prepare_complete:
            time.sleep(0.1)
        self._send_begin_decode()
        with time_meter.timer("LatencyPerToken"):    
            logits, scores = self.rag.decoding()
            self.draft_loc.put((logits, scores))
            output_ids = self.aggregate()
        time_meter.timer("LatencyPerToken").duration /= max_new_tokens
        self.stats.update(time_meter.timer("LatencyPerToken"))
        return self.rag.generator.tokenizer.decode(output_ids, skip_special_tokens=True)
    
    def aggregate(self):
        logits_loc, scores_loc = self.draft_loc.get()
        logits_rem, scores_rem = self.draft_rem.get()

        scores = torch.vstack([scores_loc, scores_rem])
        logits = torch.stack([logits_loc, logits_rem])
        logprobs = torch.nn.functional.log_softmax(     # (s_aggregate, s_sequence, s_vocab)
            logits / self.rag.generator.sampler.temperature, dim=-1)
        logprobs = logprobs.permute(1, 0, 2)            # (s_sequence, s_aggregate, s_vocab)
        logprobs.add_(scores)                           # (s_sequence, s_aggregate, s_vocab) + (s_aggregate, 1)
        logprobs = torch.logsumexp(logprobs, dim=1)
        output_ids = self.rag.generator.sampler(torch.exp(logprobs))
        return output_ids

    def _send_begin_generate(self, query: str, template: str, max_new_tokens: int):
        self.transceiver.send(Message.BEGIN_GENERATE, (query, template, max_new_tokens))
    
    def _send_draft_sequence(self, logits, scores):
        self.logger.info("Sending draft sequence.")
        self.transceiver.send(Message.DRAT_SEQUENCE, (logits, scores))
    
    def _send_prepare_complete(self):
        self.logger.info("Sending preparation complete signal.")
        self.transceiver.send(Message.PREPARE_COMPLETE, None)
    
    def _send_begin_decode(self):
        self.logger.info("Sending begin decode signal.")
        self.transceiver.send(Message.BEGIN_DECODE, None)

    def _collect_observers(self):
        return [
            self._rx_ready_for_generation,
            self._rx_begin_generate,
            self._rx_draft_sequence,
            self._rx_prepare_complete,
            self._rx_begin_decode,
            self._rx_shutdown
        ]

    def _rx_ready_for_generation(self, mtype: int, mbody: object):
        if mtype != Message.READY_FOR_GENERATION: return False
        self.ready_for_generation = True
        self.logger.info("Remote is ready for generation.")
        return True
    
    def _rx_begin_generate(self, mtype: int, mbody: object):
        if mtype != Message.BEGIN_GENERATE: return False
        query, template, max_new_tokens = mbody
        self.logger.info(f"Generating response for query: {query}")
        self.rag.retrieval_and_prefilling(query, max_new_tokens, template)
        self._send_prepare_complete()
        return True

    def _rx_prepare_complete(self, mtype: int, mbody: object):
        if mtype != Message.PREPARE_COMPLETE: return False
        self.logger.info("Remote's preparation is complete.")
        self.remote_prepare_complete = True
        return True
    
    def _rx_begin_decode(self, mtype: int, mbody: object):
        if mtype != Message.BEGIN_DECODE: return False
        self.logger.info("Received begin decode signal.")
        logits, scores = self.rag.decoding()
        self._send_draft_sequence(logits.cpu().tolist(), scores.cpu().tolist())
    
    def _rx_draft_sequence(self, mtype: int, mbody: object):
        if mtype != Message.DRAT_SEQUENCE: return False
        logits, scores = mbody
        logits = torch.as_tensor(logits, dtype=torch.float32).to(self.rag.generator.device)
        scores = torch.as_tensor(scores, dtype=torch.float32).to(self.rag.generator.device)
        self.draft_rem.put((logits, scores))
        self.logger.info("Draft sequence received.")
        return True
    
    def _rx_shutdown(self, mtype: int, mbody: object):
        if mtype != Message.SHUTDOWN: return False
        self.logger.info("Received shutdown signal.")
        self._shutdown()
        return True