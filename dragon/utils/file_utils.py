import json
import pickle
from abc import ABC, abstractmethod


class Serializer(ABC):

    @abstractmethod
    def load(self, path: str):
        pass
    
    @abstractmethod
    def dump(self, data, path: str):
        pass

    @staticmethod
    def from_suffix(suffix: str) -> 'Serializer':
        if suffix == '.pkl':
            return PickleSerializer()
        elif suffix == '.jsonl':
            return JsonLinesSerializer()
        elif suffix == '.json':
            return JsonSerializer()
        else:
            raise ValueError(f"Unsupported file suffix: {suffix}")

class JsonSerializer(Serializer):

    def load(self, path: str):
        with open(path, 'r') as f:
            return json.load(f)

    def dump(self, data, path: str):
        with open(path, 'w') as f:
            json.dump(data, f)


class PickleSerializer(Serializer):
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def dump(self, data, path: str):
        with open(path, 'wb') as f:
            pickle.dump(data, f)

class JsonLinesSerializer(Serializer):

    def load(self, path: str):
        with open(path, 'r') as f:
            return [json.loads(line) for line in f]

    def dump(self, data, path: str):
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')