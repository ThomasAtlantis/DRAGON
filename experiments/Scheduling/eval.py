import sys; sys.path.append(".")
from experiments.Scheduling.simulator import Config, EmuWorld, EmuCloud, EmuDevice
from experiments.Scheduling.simulator import acceptance_stats
from dragon.utils.meter import Statistics
from dragon.utils.mlogging import Logger
from functools import partial
import numpy as np

config = Config()
config.parse_sys_args()
config.precision = 0.01
stats = Statistics()
logger = Logger.build("Simulator", "INFO")

np.random.seed(0)
n_repeats = 50
acceptance_rate_settings = [[0.49, 0.51], [0.99, 0.01]]
acceptance_pool_cloud = [np.random.choice(
    [True, False], size=config.max_new_tokens, p=acceptance_rate_settings[np.random.choice([0, 1])])
    for _ in range(n_repeats)]
acceptance_pool_device = [np.random.choice(
    [True, False], size=config.max_new_tokens, p=acceptance_rate_settings[np.random.choice([0, 1])])
    for _ in range(n_repeats)]
# n_repeats = 1
# acceptance_pool_cloud = [np.tile(np.repeat([True, False], 10), (config.max_new_tokens + 9) // 10)]
# acceptance_pool_device = [np.random.choice(
#     [True, False], size=config.max_new_tokens, p=[0.99, 0.01])
#     for _ in range(n_repeats)]

def latency_trace(t: float, latency: float, jitter: float):
    np.random.seed(int(t * 100))
    return 0.002 + np.random.uniform(0, 0.006) + latency + np.sin(t / 10) * jitter

def evaluate_baseline(aggregate_method: str, schedule_strategy: str, latency: float):
    # logger.info(f"Evaluating Aggregator({aggregate_method}) Scheduler({schedule_strategy}) ...")
    config.method = aggregate_method
    config.scheduler.strategy = schedule_strategy
    network_latency = partial(latency_trace, latency=latency, jitter=latency/5)
    world = EmuWorld(config, network_latency)
    world.add_entities([EmuCloud(world, config), EmuDevice(world, config)])
    world.run_until(100)
    stats.update(name=f"{aggregate_method}-{schedule_strategy}", stat=world.wall_time)

def evaluate(latency: float):
    logger.info(f"Starting evaluation for latency={latency * 1000} ms ...")
    stats.new_record()
    stats.update(name="latency", stat=latency)
    
    for i in range(n_repeats):
        acceptance_stats['Cloud'] = acceptance_pool_cloud[i]
        acceptance_stats['Device'] = acceptance_pool_device[i]
        evaluate_baseline("speculative", "cloud", latency)
        evaluate_baseline("speculative", "device", latency)
        evaluate_baseline("speculative", "always", latency)
        evaluate_baseline("speculative", "random", latency)
        evaluate_baseline("speculative", "dragon", latency)
        
        evaluate_baseline("synchronized", "device", latency)
        evaluate_baseline("synchronized", "cloud", latency)
        evaluate_baseline("synchronized", "always", latency)
        evaluate_baseline("synchronized", "random", latency)
        evaluate_baseline("synchronized", "dragon", latency)

def main():
    evaluate(latency=0)
    evaluate(latency=0.1)
    evaluate(latency=0.2)
    evaluate(latency=0.3)
    evaluate(latency=0.4)
    stats.dump("scheduling_stats.json")
    logger.info("Evaluation completed.")
    logger.info("Results saved to `scheduling_stats.json`.")

if __name__ == "__main__":
    main()
