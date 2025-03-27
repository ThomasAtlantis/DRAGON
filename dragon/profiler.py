import re
import subprocess
import threading

import numpy as np
from tqdm import trange

from .utils.mlogging import Logger
from .utils.meter import Statistics, TimeMeter
from .config import DragonConfig
from .rag import Rag

logging_level = "INFO"
time_meter = TimeMeter()

class OfflineProfiler:
    def __init__(self, config: DragonConfig, rag: Rag, query: str, prompt_template: str, max_new_tokens: int):
        self.logger = Logger.build(__class__.__name__, level=logging_level)
        self.rag = rag
        self.config = config
        self.query = query
        self.prompt_template = prompt_template
        self.max_new_tokens = max_new_tokens
        self.stats = Statistics()
    
    def start(self):
        self.latency_dec_loc, self.rtt, self.jitter = self._offline_profiling()
        self.logger.info(f"latency_dec_loc: {self.latency_dec_loc}")
        if self.rtt is not None:
            self.logger.info(f"rtt: {self.rtt:>.4f} us")
            self.logger.info(f"jitter: {self.jitter:>.4f} us")

    def _offline_latency_dec_loc(self):
        self.logger.info("Measuring the latency of local decoding ...")
        start_step = -1
        for epoch in trange(self.config.profiler.n_epochs, desc="Epochs"):
            input_ids, attention_mask, scores, passages = self.rag._prepare_inputs_for_generation(
                query=self.query, prompt_template=self.prompt_template
            )
            if start_step < 0:
                start_step = input_ids.shape[-1]
            output = self.rag._generate(input_ids, attention_mask, scores, preemptable=False)
            self.stats.new_record()
            for step in trange(self.max_new_tokens, desc="Steps", leave=False):
                with time_meter.timer("latency_dec_loc"):
                    output, attention_mask = self.rag.generate(
                        output.next_token, scores, attention_mask, past_key_values=output.past_key_values,
                        preemptable=False
                    )
                self.stats.update(time_meter.timer("latency_dec_loc"))
        latency_dec_loc = np.mean([record['latency_dec_loc'] for record in self.stats.records], axis=0)
        self.stats.new_record()
        self.stats.update(name="start_step", stat=start_step)
        return latency_dec_loc

    def _offline_rtt(self, port=11111, test_time=10, msg_size=14):
        cmd = [
            "sockperf", "pp",
            "-i", self.config.trans.tx_host,
            "-p", str(port),
            "--tcp",
            "-t", str(test_time),
            "-m", str(msg_size)
        ]
        
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
            avg_latency_match = re.search(r"avg-lat=(\d+\.\d+) \(std-dev=(\d+\.\d+)\)", output)
            if avg_latency_match:
                rtt = float(avg_latency_match.group(1))
                jitter = float(avg_latency_match.group(2))
                self.stats.new_record()
                self.stats.update(name="rtt", stat=rtt)
                self.stats.update(name="jitter", stat=jitter)
                return rtt, jitter
            else:
                raise ValueError("Latency not found in output of sockperf.")
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"sockperf failed with return code {e.returncode}.")
            self.logger.error(e.output)
            return None

    def _offline_profiling(self):
        self.logger.info("Offline profiling ...")
        latency_dec_loc = self._offline_latency_dec_loc()
        rtt, jitter = self._offline_rtt() if self.config.trans.rank == 1 else (None, None)
        return latency_dec_loc, rtt, jitter
