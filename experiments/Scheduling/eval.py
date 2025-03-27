from abc import abstractmethod
import json
from typing import List
import numpy as np
from dataclasses import dataclass
from dragon.utils.configure import Configure, Field


class Rank:
    CLOUD = 0
    DEVICE = 1
    

class Config(Configure):
    stats           = Field(str, default="outputs/Latency-20250326224732/stats.json")
    offline_profile = Field(str, default="outputs/Latency-20250326224732/profile.json")
    online_profile  = Field(str, default="outputs/Latency-20250326224732/decoder_stats.json")
    method          = Field(str, default="speculative")
    max_draft_tokens = Field(int, default=3)
    max_new_tokens = Field(int, default=10)

class Observer:

    def __init__(self):
        self.running = True
    
    @abstractmethod
    def update(self, world):
        raise NotImplementedError
    
class Network(Observer):

    def __init__(self, config: Config):
        super().__init__()
        self.tasks = []
        self.observers = {}

    def add_observers(self, observers):
        self.observers.update({
            observer.rank: observer for observer in observers
        })

    def add_task(self, title, src_id, dst_id, elapsed_time, message):
        self.tasks.append({
            "title": title,
            "src_id": src_id,
            "dst_id": dst_id,
            "elapsed_time": elapsed_time,
            "latency": 0.7,
            "message": message,
            "complete": False,
        })

    def update(self, world):
        for task in self.tasks:
            if task['complete']: continue
            if task['elapsed_time'] + world.tick > task['latency']:
                world.records.append({
                    "title": task['title'],
                    "src_id": task['src_id'],
                    "dst_id": task['dst_id'],
                    "beg_time": world.wall_time - task['elapsed_time'],
                    "duration": task['latency']
                })
                self.observers[task['dst_id']].notify(
                    task['src_id'], task['title'], task['message'], world.wall_time + task['latency'] - task['elapsed_time']
                )
                task['complete'] = True
            else:
                task["elapsed_time"] += world.tick

class EmuWorld:

    def __init__(self, config: Config, precision=1e-6):
        self.tick = precision
        self.wall_time = 0
        self.observers: List[Observer] = []
        self.network = Network(config)
        self.records = []
    
    def update(self):
        self.wall_time += self.tick
    
    def add_observers(self, observers: List[Observer]):
        self.observers.extend(observers)
        self.network.add_observers(observers)
    
    def is_running(self):
        return any([o.running for o in self.observers])

    def run_until(self, end_time: float):    
        while self.wall_time < end_time:
            if not self.is_running():
                break
            self.network.update(self)
            for observer in self.observers:
                observer.update(self)
            self.update()

@dataclass
class SystemProfile:
    rtt: float
    latency_dec_rem: float
    latency_dec_loc: float
    acceptance_rate_rem: float
    acceptance_rate_loc: float

def in_range(x, r):
    return r[0] < x <= r[1]

def strategy(profile: SystemProfile):
    delta_z: float
    if profile.latency_dec_loc == 0:
        delta_z = (1 - profile.acceptance_rate_rem) * profile.rtt
    elif in_range(profile.latency_dec_loc, (
            0, 
            profile.latency_dec_rem - profile.rtt,
        )):
        delta_z = (1 - profile.acceptance_rate_rem) * profile.rtt
    elif in_range(profile.latency_dec_loc, (
            profile.latency_dec_rem - profile.rtt, 
            profile.latency_dec_rem,
        )):
        delta_z = (1 - profile.acceptance_rate_loc) * (profile.latency_dec_rem - profile.latency_dec_loc) + \
            (profile.acceptance_rate_loc - profile.acceptance_rate_rem) * profile.rtt
    elif in_range(profile.latency_dec_loc, (
            profile.latency_dec_rem,
            profile.latency_dec_rem + profile.rtt,
        )):
        delta_z = (1 - profile.acceptance_rate_rem) * (profile.latency_dec_rem - profile.latency_dec_loc) + \
            (profile.acceptance_rate_loc - profile.acceptance_rate_rem) * profile.rtt
    else:
        delta_z = (profile.acceptance_rate_loc - 1) * profile.rtt
    return delta_z

class EmuDecoder(Observer):

    def __init__(self, rank: int, latency: float, draft_tokens: list, max_draft_tokens: int, max_new_tokens: int):
        super().__init__()
        self.latency = latency
        self.elapsed_time = 0
        self.current_step = 0
        self.max_step = min(len(latency), max_new_tokens)
        self.rank = rank
        self.draft_tokens = draft_tokens
        self.max_draft_tokens = max_draft_tokens
        self.version = 0

    def current_task(self):
        if len(self.draft_tokens) == self.max_draft_tokens or self.current_step == self.max_step:
            return 0
        return self.latency[self.current_step]
    
    def update(self, world: EmuWorld):
        current_task = self.current_task()
        if current_task == 0: 
            self.running = False
            return
        self.running = True
        if self.elapsed_time + world.tick > current_task:
            world.records.append({
                "title": "Decode",
                "beg_time": world.wall_time - self.elapsed_time,
                "end_time": world.wall_time + current_task - self.elapsed_time,
                'duration': current_task,
                "meta": {
                    "step": self.current_step,
                    "rank": self.rank
                }
            })
            dst_id = Rank.CLOUD if self.rank == Rank.DEVICE else Rank.DEVICE
            self.elapsed_time = self.elapsed_time + world.tick - current_task
            
            draft_token = {
                "token": (self.rank + 1) * self.current_step,
                "time": world.wall_time + current_task - self.elapsed_time,
                "step": self.current_step,
                'version': self.version
            }
            self.draft_tokens.append(draft_token)
            world.network.add_task(
                title="DraftToken", src_id=self.rank, dst_id=dst_id, 
                elapsed_time=self.elapsed_time, message=draft_token)
            
            self.current_step += 1
        else:            
            self.elapsed_time += world.tick
        
class EmuDecoderDevice(EmuDecoder):

    def __init__(self, config: Config, draft_tokens: list):
        with open(config.online_profile) as f:
            stats = json.load(f)[0]
        super().__init__(rank=Rank.DEVICE, latency=stats['latency_dec_loc'], 
                         draft_tokens=draft_tokens, max_draft_tokens=config.max_draft_tokens, 
                         max_new_tokens=config.max_new_tokens)

class EmuDecoderCloud(EmuDecoder):

    def __init__(self, config: Config, draft_tokens: list):
        with open(config.offline_profile) as f:
            stats = json.load(f)
        latency = np.mean([stat['latency_dec_rem'] for stat in stats[:-1]])
        with open(config.online_profile) as f:
            stats = json.load(f)[0]
        latency = [latency] * len(stats['latency_dec_loc'])
        super().__init__(rank=Rank.CLOUD, latency=latency, draft_tokens=draft_tokens, 
                         max_draft_tokens=config.max_draft_tokens, max_new_tokens=config.max_new_tokens)


class EmuAggregator(Observer):

    def __init__(self, config: Config, draft_tokens_loc: list, draft_tokens_rem: list, target_tokens: list, rank: int, method):
        super().__init__()
        with open(config.stats) as f:
            stats = json.load(f)[0]
        self.acceptance_loc = stats['AcceptanceLoc']
        self.acceptance_rem = stats['AcceptanceRem']
        self.draft_tokens_loc = draft_tokens_loc
        self.draft_tokens_rem = draft_tokens_rem
        self.version_loc = 0
        self.version_rem = 0
        self.target_tokens = target_tokens
        self.where = rank
        self.method = method
    
    def update(self, world: EmuWorld):
        while len(self.draft_tokens_loc) > 0 and len(self.draft_tokens_rem) > 0:

            while len(self.draft_tokens_loc) > 0 and (
                    self.draft_tokens_loc[0]['version'] < self.version_loc or \
                    self.draft_tokens_loc[0]['step'] != len(self.target_tokens)
                ):
                self.draft_tokens_loc.pop(0)
            if len(self.draft_tokens_loc) > 0: 
                draft_loc = self.draft_tokens_loc[0]
            else:
                continue                
        
            while len(self.draft_tokens_rem) > 0 and (
                    self.draft_tokens_rem[0]['version'] < self.version_rem or \
                    self.draft_tokens_rem[0]['step'] != len(self.target_tokens)
                ):
                self.draft_tokens_rem.pop(0)
            if len(self.draft_tokens_rem) > 0: 
                draft_rem = self.draft_tokens_rem[0]
            else:
                continue

            target_token = draft_loc['token'] + draft_rem['token']
            dst_id = Rank.CLOUD if self.where == Rank.DEVICE else Rank.DEVICE
            accept_loc = self.acceptance_loc[len(self.target_tokens)] if self.method == "speculative" else False
            accept_rem = self.acceptance_rem[len(self.target_tokens)] if self.method == "speculative" else False

            if not accept_loc: self.version_loc += 1
            if not accept_rem: self.version_rem += 1

            step = len(self.target_tokens) + 1
            world.network.observers[self.where].notify(
                self.where, "TargetToken", {
                    "token": target_token,
                    "step": step,
                    "accept_loc": accept_loc,
                    "accept_rem": accept_rem
                }, world.wall_time
            )
            world.network.add_task(
                title="TargetToken", src_id=self.where, dst_id=dst_id, 
                elapsed_time=max(0, world.wall_time - max(draft_loc['time'], draft_rem['time'])), 
                message={
                    "token": target_token,
                    "step": step,
                    "accept_loc": accept_rem,
                    "accept_rem": accept_loc
                })

class Node(Observer):

    def __init__(self, rank):
        super().__init__()
        self.rank = rank
        self.draft_tokens_loc = []
        self.draft_tokens_rem = []
        self.target_tokens = []
        self.decoder = None
        self.aggregator = None

    def notify(self, src_id, title, message, time):
        if title == "DraftToken":
            message['time'] = time
            self.draft_tokens_rem.append(message)
            return True
        if title == "TargetToken":
            if not message['accept_loc']:
                self.decoder.current_step = message['step']
                self.decoder.elapsed_time = 0
                self.decoder.version += 1
            while len(self.draft_tokens_loc) > 0 and (
                self.draft_tokens_loc[0]['version'] < self.decoder.version or \
                self.draft_tokens_loc[0]['step'] != message['step']
            ):
                self.draft_tokens_loc.pop(0)
            
            if not message['accept_rem']:
                self.draft_tokens_rem.clear()
            else:
                self.draft_tokens_rem.pop(0)
            self.target_tokens.append(message['token'])
            return True

class EmuDevice(Node):
    def __init__(self, config: Config):
        super().__init__(Rank.DEVICE)
        self.decoder = EmuDecoderDevice(config, self.draft_tokens_loc)
        self.aggregator = EmuAggregator(config, self.draft_tokens_loc, self.draft_tokens_rem, self.target_tokens, self.rank, config.method)
    
    def update(self, world):
        self.decoder.update(world)
        self.aggregator.update(world)
        self.running = len(self.target_tokens) < self.decoder.max_step

class EmuCloud(Node):
    def __init__(self, config: Config):
        super().__init__(Rank.CLOUD)
        self.decoder = EmuDecoderCloud(config, self.draft_tokens_loc)
        self.aggregator = None
    
    def update(self, world):
        self.decoder.update(world)
        self.running = len(self.target_tokens) < self.decoder.max_step
    