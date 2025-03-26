import re
import subprocess
import threading

import numpy as np

from .utils.mlogging import Logger
from .utils.meter import Statistics, TimeMeter
from .config import DragonConfig
from .rag import Rag

logging_level = "INFO"
time_meter = TimeMeter()

class Profiler(threading.Thread):
    def __init__(self, config: DragonConfig, rag: Rag, query: str, prompt_template: str, max_new_tokens: int):
        super().__init__()
        self.logger = Logger.build(__class__.__name__, level=logging_level)
        self.rag = rag
        self.config = config
        self.query = query
        self.prompt_template = prompt_template
        self.max_new_tokens = max_new_tokens
        self.stats = Statistics()
        self.latency_dec_loc, self.rtt = self._offline_profiling()
        self.logger.info(f"latency_dec_loc: {self.latency_dec_loc}")
        self.logger.info(f"rtt: {self.rtt}")

    def _offline_latency_dec_loc(self):
        for _ in range(10):
            self.stats.new_record()
            input_ids, attention_mask, scores, passages = self.rag._prepare_inputs_for_generation(
                query=self.query, prompt_template=self.prompt_template
            )
            output = self.rag._generate(input_ids, attention_mask, scores)
            for step in range(self.max_new_tokens):
                with time_meter.timer("latency_dec_loc"):
                    output, attention_mask = self.rag.generate(
                        output.next_token, scores, attention_mask, past_key_values=output.past_key_values,
                        preemptable=False
                    )
                self.stats.update(time_meter.timer("latency_dec_loc"))
        latency_dec_loc = np.mean([record['latency_dec_loc'] for record in self.stats.records], axis=0)
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
            print(output)
            avg_latency_match = re.search(r"avg-lat=(\d+\.\d+)", output)
            if avg_latency_match:
                rtt = float(avg_latency_match.group(1))
                self.stats.new_record()
                self.stats.update({"rtt": rtt})
                return rtt
            else:
                raise ValueError("Latency not found in output of sockperf.")
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"sockperf failed with return code {e.returncode}.")
            self.logger.error(e.output)
            return None

    def _offline_profiling(self):
        self.logger.info("Offline profiling ...")
        latency_dec_loc = self._offline_latency_dec_loc()
        rtt = self._offline_rtt() if self.config.trans.rank == 1 else None
        return latency_dec_loc, rtt

    def run(self):
        pass