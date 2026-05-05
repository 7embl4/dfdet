import torch
import numpy as np

from PIL import Image
from pathlib import Path

from src.dataset import BaseDataset
from src.utils.io import ROOT_PATH, write_json, read_json


class ImageDataset(BaseDataset):
    def __init__(
        self, 
        data_dir,
        part="train",
        val_size=0.1,
        *args,
        **kwargs
    ):
        self.data_dir = ROOT_PATH / data_dir
        self.part = part
        self.val_size = val_size
        self.formats = (".jpg", ".jpeg", ".png")

        index = self._load_or_create_index()
        super().__init__(index, *args, **kwargs)

    def _load_or_create_index(self):
        index = None
        filename = "index_" + self.part + f"_{self.val_size}.json"
        if (self.data_dir / filename).exists():
            index = read_json(self.data_dir / filename)
        else:
            index = self._create_index()
            write_json(index, self.data_dir / filename)
        return index

    def _create_index(self):
        index = []
        print(f"Creating {self.part} index")

        # fake videos
        for fake_type in (self.data_dir / "fake").iterdir():
            fake_videos = []
            for frames_path in fake_type.iterdir():
                fake_videos.append({
                    "frames_path": str(frames_path),
                    "target": 1
                })
        
            val_size = int(len(fake_videos) * self.val_size)
            if self.part == "train":
                index.extend(fake_videos[:-val_size])
            else:
                index.extend(fake_videos[-val_size:])

        # real videos
        real_videos = []
        for frames_path in (self.data_dir / "real").iterdir():
            real_videos.append({
                "frames_path": str(frames_path),
                "target": 0
            })
        
        val_size = int(len(real_videos) * self.val_size)
        if self.part == "train":
            index.extend(real_videos[:-val_size])
        else:
            index.extend(real_videos[-val_size:])

        return index
    
    def __getitem__(self, idx):
        instance_data = dict(self._index[idx])
        instance_data["frames"] = np.stack([
            Image.open(str(frame)) for frame in Path(instance_data["frames_path"]).iterdir()
        ])
        instance_data["target"] = torch.tensor(instance_data["target"])

        instance_data = self.apply_instance_transforms(instance_data)
        instance_data = self.apply_augmentations(instance_data)

        return instance_data
  