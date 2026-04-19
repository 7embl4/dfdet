import torch


def collate_fn(dataset_items: list[dict]):
    batch = {k: [] for k in dataset_items[0].keys()}
    for item in dataset_items:
        for k, v in item.items():
            batch[k].append(v)
        
    for k, v in batch.items():
        if isinstance(v[0], torch.Tensor):
            batch[k] = torch.stack(v)
    
    return batch
