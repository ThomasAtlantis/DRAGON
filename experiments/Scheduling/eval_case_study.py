import json
import pickle
import sys; sys.path.append(".")
from experiments.Scheduling.simulator import Config, EmuWorld, EmuCloud, EmuDevice
from experiments.Scheduling.simulator import acceptance_stats
import numpy as np

def latency_trace(t: float, latency: float = 0.5, jitter: float = 0.1):
    np.random.seed(int(t * 100))
    return 0.002 + np.random.uniform(0, 0.006) + latency + np.sin(t / 5) * jitter

if __name__ == "__main__":
    config = Config()
    config.max_draft_tokens = 5
    config.max_new_tokens = 15
    config.precision = 0.01
    config.method = "speculative"
    config.scheduler.strategy = "dragon"

    with open("outputs/Latency-20250330174023/stats.json") as f:
        stats = json.load(f)
    acceptance_stats['Cloud'] = stats[0]['AcceptanceLoc'][: config.max_new_tokens]
    acceptance_stats['Device'] = stats[0]['AcceptanceRem'][: config.max_new_tokens]


    np.random.seed(42)
    world = EmuWorld(config, latency_trace)
    world.add_entities([EmuCloud(world, config), EmuDevice(world, config)])
    world.run_until(100)

    with open("outputs/scheduling_case.pkl", "wb") as f:
        pickle.dump([world, acceptance_stats], f)
