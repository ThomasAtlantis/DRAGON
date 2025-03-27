from abc import abstractmethod
import json
from typing import List, Union
import numpy as np
from dataclasses import dataclass
from dragon.utils.configure import Configure, Field

class Config(Configure):
    stats           = Field(str, default="outputs/Latency-20250326224732/stats.json")
    offline_profile = Field(str, default="outputs/Latency-20250326224732/profile.json")
    online_profile  = Field(str, default="outputs/Latency-20250326224732/decoder_stats.json")
    method          = Field(str, default="speculative")
    max_draft_tokens = Field(int, default=3)
    max_new_tokens = Field(int, default=10)

class Rank:
    CLOUD = 0
    DEVICE = 1

class Message:
    TARGET = "TargetToken"
    DRAFT = "DraftToken"

@dataclass
class TransmissionTask:
    title: str
    src_id: int
    dst_id: int
    message: str
    elapsed_time: float
    latency: float = 0.7
    complete: bool = False

@dataclass
class WorldRecord:
    title: str
    beg_time: float
    duration: float
    meta: dict

@dataclass
class DraftItem:
    token: int
    time: float
    step: int
    version: int

@dataclass
class TargetItem:
    token: int
    step: int
    accept_loc: bool
    accept_rem: bool

@dataclass
class SystemProfile:
    rtt: float
    latency_dec_rem: float
    latency_dec_loc: float
    acceptance_rate_rem: float
    acceptance_rate_loc: float

class Entity:

    def __init__(self, world):
        self.running = True
        self.world = world
    
    @abstractmethod
    def update(self):
        raise NotImplementedError
    
class Network(Entity):

    def __init__(self, world):
        super().__init__(world)
        self.tasks: List[TransmissionTask] = []
        self.entities: dict[Entity] = {}

    def add_entities(self, entities):
        self.entities.update({
            entity.rank: entity for entity in entities
        })

    def add_task(self, task: TransmissionTask):
        self.tasks.append(task)

    def update(self):
        for task in self.tasks:
            if task.complete: continue
            if task.elapsed_time + self.world.tick > task.latency:
                self.world.add_record(WorldRecord(
                    title=task.title,
                    beg_time=self.world.wall_time - task.elapsed_time,
                    duration=task.latency,
                    meta={
                        "src_id": task.src_id,
                        "dst_id": task.dst_id,
                    }
                ))
                self.entities[task.dst_id].notify(
                    task.title, task.message, self.world.wall_time + task.latency - task.elapsed_time
                )
                task.complete = True
            else:
                task.elapsed_time += self.world.tick
    
class EmuWorld:

    def __init__(self, precision=1e-6):
        self.tick = precision
        self.wall_time = 0
        self.network = Network(self)
        self.entities: List[Entity] = []
        self.records: List[WorldRecord] = []
    
    def get_entity_by_id(self, entity_id: int) -> Entity:
        for entity in self.entities:
            if entity.rank == entity_id:
                return entity
        return None

    def update(self):
        self.wall_time += self.tick
    
    def add_entities(self, entities: List[Entity]):
        self.entities.extend(entities)
        self.network.add_entities(entities)

    def add_record(self, record: WorldRecord):
        self.records.append(record)
    
    def is_running(self):
        return any([o.running for o in self.entities])

    def run_until(self, end_time: float):    
        while self.wall_time < end_time:
            if not self.is_running():
                break
            self.network.update()
            for entity in self.entities:
                entity.update()
            self.update()

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

class EmuDecoder(Entity):

    def __init__(self, world: EmuWorld, config: Config, rank: int, latency: float, draft_tokens: list):
        super().__init__(world)
        self.rank = rank
        self.latency = latency
        self.elapsed_time = 0
        self.current_step = 0
        self.max_step = min(len(latency), config.max_new_tokens)
        self.draft_tokens = draft_tokens
        self.max_draft_tokens = config.max_draft_tokens
        self.version = 0
        self.dst_id = Rank.CLOUD if self.rank == Rank.DEVICE else Rank.DEVICE
    
    def roll_back(self, step: int):
        self.current_step = step
        self.elapsed_time = 0
        self.version += 1

    def current_task(self):
        if len(self.draft_tokens) == self.max_draft_tokens or \
            self.current_step == self.max_step:
            return 0
        return self.latency[self.current_step]

    def synchronize_draft(self, draft_item: DraftItem):
        self.draft_tokens.append(draft_item)
        self.world.network.add_task(
            TransmissionTask(
                title=Message.DRAFT,
                src_id=self.rank,
                dst_id=self.dst_id,
                message=draft_item,
                elapsed_time=self.elapsed_time,
            )
        )
    
    def record(self, current_task: float):
        self.world.add_record(WorldRecord(
            title="Decode",
            beg_time=self.world.wall_time - self.elapsed_time,
            duration=current_task,
            meta={
                "step": self.current_step,
                "rank": self.rank
            }
        ))
    
    def update(self):
        current_task = self.current_task()
        if current_task == 0: 
            self.running = False
            return
        self.running = True
        if self.elapsed_time + self.world.tick > current_task:
            self.record(current_task)
            self.elapsed_time = self.elapsed_time + self.world.tick - current_task
            self.synchronize_draft(DraftItem(
                token=(self.rank + 1) * self.current_step,
                time=self.world.wall_time + current_task - self.elapsed_time,
                step=self.current_step,
                version=self.version
            ))            
            self.current_step += 1
        else:            
            self.elapsed_time += self.world.tick
        
class EmuDecoderDevice(EmuDecoder):

    def __init__(self, world: EmuWorld, config: Config, draft_tokens: list):
        with open(config.online_profile) as f:
            latency = json.load(f)[0]['latency_dec_loc']
        super().__init__(world, config, rank=Rank.DEVICE, latency=latency, draft_tokens=draft_tokens)

class EmuDecoderCloud(EmuDecoder):

    def __init__(self, world: EmuWorld, config: Config, draft_tokens: list):
        with open(config.offline_profile) as f:
            stats = json.load(f)
        latency = np.mean([stat['latency_dec_rem'] for stat in stats[:-1]])
        with open(config.online_profile) as f:
            n_steps = len(json.load(f)[0]['latency_dec_loc'])
        latency = [latency] * n_steps
        super().__init__(world, config, rank=Rank.CLOUD, latency=latency, draft_tokens=draft_tokens)


class EmuAggregator(Entity):

    def __init__(self, world: EmuWorld, config: Config, draft_tokens_loc: list, draft_tokens_rem: list, target_tokens: list, rank: int):
        super().__init__(world)
        with open(config.stats) as f:
            stats = json.load(f)[0]
        self.acceptance_loc = stats['AcceptanceLoc']
        self.acceptance_rem = stats['AcceptanceRem']
        self.draft_tokens_loc: List[DraftItem] = draft_tokens_loc
        self.draft_tokens_rem: List[DraftItem] = draft_tokens_rem
        self.target_tokens: List[TargetItem] = target_tokens
        self.method = config.method
        self.version_loc = self.version_rem = 0
        self.where = rank
        self.dst_id = Rank.CLOUD if self.where == Rank.DEVICE else Rank.DEVICE

    def _get_draft(self, draft_tokens: List[DraftItem], version: int):
        while len(draft_tokens) > 0 and (
                draft_tokens[0].version < version or \
                draft_tokens[0].step != len(self.target_tokens)
            ):
            draft_tokens.pop(0)
        return draft_tokens[0] if len(draft_tokens) > 0 else None

    def _aggregate(self, draft_token_loc, draft_token_rem):
        if self.method == "speculative":
            accept_loc = self.acceptance_loc[len(self.target_tokens)]
            accept_rem = self.acceptance_rem[len(self.target_tokens)]
        else:
            accept_loc = accept_rem = False
        target_item = TargetItem(
            token=draft_token_loc + draft_token_rem,
            accept_loc=accept_loc, accept_rem=accept_rem,
            step=len(self.target_tokens) + 1)
        return target_item
    
    def _update_version(self, target_item: TargetItem):
        if not target_item.accept_loc: self.version_loc += 1
        if not target_item.accept_rem: self.version_rem += 1
    
    def _adjust_queues(self, target_item: TargetItem, elapsed_time: float):
        self.world.get_entity_by_id(self.where).notify(
            Message.TARGET, target_item, self.world.wall_time
        )
        target_item.accept_loc, target_item.accept_rem = target_item.accept_rem, target_item.accept_loc
        self.world.network.add_task(TransmissionTask(
            title=Message.TARGET, src_id=self.where, dst_id=self.dst_id,
            elapsed_time=elapsed_time, message=target_item
        ))

    def update(self):
        while len(self.draft_tokens_loc) > 0 and len(self.draft_tokens_rem) > 0:
            draft_loc = self._get_draft(self.draft_tokens_loc, self.version_loc)
            if draft_loc is None: continue
            draft_rem = self._get_draft(self.draft_tokens_rem, self.version_rem)
            if draft_rem is None: continue
            target_item = self._aggregate(draft_loc.token, draft_rem.token)
            self._update_version(target_item)
            # transmission should start `elapsed_time` ago
            elapsed_time = max(0, self.world.wall_time - max(draft_loc.time, draft_rem.time))
            self._adjust_queues(target_item, elapsed_time)
            

class Node(Entity):

    def __init__(self, world: EmuWorld, rank):
        super().__init__(world)
        self.rank = rank
        self.draft_tokens_loc: List[DraftItem] = []
        self.draft_tokens_rem: List[DraftItem] = []
        self.target_tokens: List[TargetItem] = []
        self.decoder: EmuDecoder = None
        self.aggregator: EmuAggregator = None
    
    def _adjust_queues_loc(self, accept_loc: bool, step: int):
        if not accept_loc:
            self.decoder.roll_back(step)
        while len(self.draft_tokens_loc) > 0 and (
            self.draft_tokens_loc[0].version < self.decoder.version or \
            self.draft_tokens_loc[0].step != step
        ):
            self.draft_tokens_loc.pop(0)
    
    def _adjust_queues_rem(self, accept_rem: bool, step: int):
        if not accept_rem:
            self.draft_tokens_rem.clear()
        else:
            self.draft_tokens_rem.pop(0)

    def notify(self, title: str, message: Union[DraftItem, TargetItem], time: float):
        if title == Message.DRAFT:
            message.time = time
            self.draft_tokens_rem.append(message)
            return True
        if title == Message.TARGET:
            self._adjust_queues_loc(message.accept_loc, message.step)
            self._adjust_queues_rem(message.accept_rem, message.step)
            self.target_tokens.append(message.token)
            return True

class EmuDevice(Node):
    def __init__(self, world: EmuWorld, config: Config):
        super().__init__(world, Rank.DEVICE)
        self.decoder = EmuDecoderDevice(world, config, self.draft_tokens_loc)
        self.aggregator = EmuAggregator(world, config, self.draft_tokens_loc, self.draft_tokens_rem, self.target_tokens, self.rank)
    
    def update(self):
        self.decoder.update()
        self.aggregator.update()
        self.running = len(self.target_tokens) < self.decoder.max_step

class EmuCloud(Node):
    def __init__(self, world: EmuWorld, config: Config):
        super().__init__(world, Rank.CLOUD)
        self.decoder = EmuDecoderCloud(world, config, self.draft_tokens_loc)
        self.aggregator = None
    
    def update(self):
        self.decoder.update()
        self.running = len(self.target_tokens) < self.decoder.max_step
    