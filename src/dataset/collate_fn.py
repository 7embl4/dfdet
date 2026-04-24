import torch
import random


def collate_fn(dataset_items: list[dict]):
    """
    Converts dataset items into a batch.

    Args:
        dataset_items (list[dict]): list of dataset items from `dataset.__getitem__`.
    """
    batch = {k: [] for k in dataset_items[0].keys()}
    
    valid = [
        i for i in range(len(dataset_items)) if dataset_items[i]["frames"] is not None
    ]
    dataset_items = [
        dataset_items[j] if j in valid else dataset_items[random.choice(valid)] 
        for j in range(len(dataset_items))
    ]

    # put all items in arrays
    for item in dataset_items:
        for k, v in item.items():
            batch[k].append(v)
    
    # stack tensors
    for k, v in batch.items():
        if isinstance(v[0], torch.Tensor):
            batch[k] = torch.stack(v)
    
    return batch
