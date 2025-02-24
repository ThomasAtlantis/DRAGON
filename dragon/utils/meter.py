import time
from contextlib import AbstractContextManager

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
    
if __name__ == "__main__":
    time_meter = TimeMeter()
    with time_meter.timer("test"):
        time.sleep(2)
        with time_meter.timer("test2"):
            time.sleep(1)
    
    print(time_meter.timer("test2"))
    print(time_meter.timer("test"))