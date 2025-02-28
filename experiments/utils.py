from dragon.utils.cache import file_cache
from pathlib import Path


def load_dataset(repo_id, dataset_id, split, cache_path=".cache"):
    @file_cache(Path(cache_path, dataset_id, "test_data.pkl"))
    def wrapper():
        import datasets
        dataset = datasets.load_dataset(repo_id, dataset_id, split=split)
        return dataset
    return wrapper()