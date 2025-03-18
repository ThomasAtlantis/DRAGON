import json
import time
from contextlib import AbstractContextManager
from typing import List


class TimeMeter:

    class Timer(AbstractContextManager):

        def __init__(self, name):
            self.start_time = None
            self.duration = None
            self.name = name

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *exc):
            self.duration = time.perf_counter() - self.start_time

        def __repr__(self):
            return f"{self.name}({self.duration:.6f}s)"
    
    def __init__(self):
        self.timers = {}
    
    def timer(self, name) -> Timer:
        if name not in self.timers:
            self.timers[name] = self.Timer(name)
        return self.timers[name]


class Statistics:
    
    def __init__(self):
        self.records: List[dict] = []

    def new_record(self):
        self.records.append({})
    
    def update(self, timer: TimeMeter.Timer = None, name: str = None, stat: float = None):
        if timer is not None:
            name = timer.name
            stat = timer.duration
        self.records[-1].setdefault(name, []).append(stat)
    
    def __or__(self, other: "Statistics") -> "Statistics":
        new_stats = Statistics()
        for record1, record2 in zip(self.records, other.records):
            new_stats.records.append(record1 | record2)
        return new_stats

    def dump(self, file_path):
        with open(file_path, "w") as f:
            json.dump(self.records, f)
    
if __name__ == "__main__":
    time_meter = TimeMeter()
    with time_meter.timer("test"):
        time.sleep(2)
        with time_meter.timer("test2"):
            time.sleep(1)
    
    print(time_meter.timer("test2"))
    print(time_meter.timer("test"))