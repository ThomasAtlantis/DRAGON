import pickle
from pathlib import Path

from .mlogging import Logger
from .file_utils import Serializer

logger = Logger.build(__name__, level="INFO")

class Cache:
    # TODO: implement API to support redis cache
    def __init__(self, name: str, load_cache, dump_cache, cache_dir=".cache"):
        self.cache = {}
        self.file = Path(cache_dir) / f"{name}.pkl"
        self.load_cache = load_cache
        self.dump_cache = dump_cache
        if self.load_cache:
            if not self.file.exists():
                self.file.touch()
            else:
                with open(self.file, "rb") as f:
                    self.cache = pickle.load(f)
    
    def set(self, key, value):
        self.cache[key] = value

    def get(self, key):
        return self.cache.get(key, None)
    
    def __contains__(self, key):
        return key in self.cache
    
    def __del__(self):
        if self.dump_cache:
            with open(self.file, "wb") as f:
                pickle.dump(self.cache, f)


def file_cache(path):
    """
    A decorator that caches the result of a function to a file.
    """
    serializer = Serializer.from_suffix(Path(path).suffix)
    def decorator(func):
        def wrapper(*args, **kwargs):
            if Path(path).exists():
                logger.info(f"Load data from cache@{path}.")
                return serializer.load(path)
            else:
                result = func(*args, **kwargs)
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                serializer.dump(result, path)
                logger.info(f"Dump data into cache@{path}.")
                return result
        return wrapper
    return decorator


if __name__ == "__main__":
    @file_cache("test.pkl")
    def some_function():
        print("First time running the function.")
        return b"This is the cached result."
    
    print(some_function())
