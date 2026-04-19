import random
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
        self, index, limit=None, shuffle_index=True, instance_transforms=None, augmentations=None
    ):
        self._assert_index_is_valid(index)
        index = self._shuffle_and_limit_index(index, shuffle_index, limit)
        self._index: list[dict] = index

        self.instance_transforms = instance_transforms
        self.augmentations = augmentations

    def __getitem__(self, idx: int):
        raise NotImplementedError()

    def __len__(self):
        return len(self._index)
    
    def apply_instance_transforms(self, instance_data: dict):
        if self.instance_transforms:
            for transform_name in self.instance_transforms.keys():
                instance_data[transform_name] = self.instance_transforms[
                    transform_name
                ](instance_data[transform_name])
        return instance_data

    def apply_augmentations(self, instance_data: dict):
        if self.augmentations:
            for transform_name in self.augmentations.keys():
                instance_data[transform_name] = self.augmentations[
                    transform_name
                ](instance_data[transform_name])
        return instance_data

    def _assert_index_is_valid(self, index: list[dict]):
        for entry in index:
            assert "target" in entry, (
                "Each dataset item should include field `target`"
            )

    def _shuffle_and_limit_index(self, index: list[dict], shuffle_index: bool, limit: int):
        if shuffle_index:
            random.shuffle(index)
        if limit:
            index = index[:limit]
        
        return index
