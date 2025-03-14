
from queue import Queue
import threading

from .rag import Rag
from .generator import CausalOutput
from .queues import DraftItem
from .transceiver import BaseTransceiver
from .utils.mlogging import Logger


logging_level = "INFO"

class Decoder(threading.Thread):
    
    def __init__(self, rag: Rag, transceiver: BaseTransceiver, target_tokens: Queue, query: str, prompt_template: str, n_steps: int):
        threading.Thread.__init__(self, name=__class__.__name__)
        self.logger = Logger.build(__class__.__name__, logging_level)
        self.rag = rag
        self.transceiver = transceiver
        self.query, self.template, self.n_steps = query, prompt_template, n_steps
        self.target_tokens = target_tokens
        self.logger.info("Decoder initialized.")

        self.input_ids, self.attention_mask, self.scores, self.passages = self.rag._prepare_inputs_for_generation(self.query, self.template)
        self.context_length = self.input_ids.shape[1]
        self.target_tokens.queue.clear()
        self.rag.generator.preempt_event.clear()

        self.output_ids = []
        self.step = 0

    def prefilling(self) -> CausalOutput:
        output = self.rag._generate(self.input_ids, self.attention_mask, self.scores)
        self._synchronize_output_to_remote(output)
        return output
    
    def _scroll_back(self, output: CausalOutput) -> CausalOutput:
        # scroll back step
        self.step = self.target_tokens.qsize() - 1
        self.logger.debug(f"Scrolling back to step {self.step}.")

        # scroll back input_ids
        output.next_token = self.target_tokens.queue[-1]
        
        # scroll back attention_mask
        cur_seq_len = self.context_length + self.step
        self.attention_mask = self.attention_mask[:, : cur_seq_len]

        # TODO: scroll back the scores
        # output.weight = ...

        # scroll back key valuse cache
        output.past_key_values = list(output.past_key_values)
        for i, _ in enumerate(output.past_key_values):
            output.past_key_values[i] = list(output.past_key_values[i])
            output.past_key_values[i][0] = output.past_key_values[i][0][..., : cur_seq_len, :]
            output.past_key_values[i][1] = output.past_key_values[i][1][..., : cur_seq_len, :]
            output.past_key_values[i] = tuple(output.past_key_values[i])
        output.past_key_values = tuple(output.past_key_values)

        return output
    
    def _synchronize_output_to_remote(self, output: CausalOutput):
        self.transceiver._send_draft_token(DraftItem(
            token=output.next_token, logprobs=output.logprobs, 
            weight=output.weight, step=self.step
        ))
    
    def decoding(self, output: CausalOutput):
        while self.target_tokens.qsize() < self.n_steps:
            temp_output, temp_attention_mask = self.rag.generate(
                output.next_token, self.scores, self.attention_mask, past_key_values=output.past_key_values)
            if not temp_output:
                output = self._scroll_back(output)
            else:
                self.step += 1
                output, self.attention_mask = temp_output, temp_attention_mask
                self._synchronize_output_to_remote(output)

    def run(self):
        self.decoding(self.prefilling())
        self.rag.generator.preempt_event.set()
        self.output_ids = [self.target_tokens.get() for _ in range(self.n_steps)]
        self.logger.debug("Generation complete.")

