import torch
import numpy as np

from PIL import Image
from pathlib import Path

from src.dataset import BaseDataset
from src.utils.io import ROOT_PATH, write_json, read_json


class ImageDataset(BaseDataset):
    def __init__(
        self, 
        data_dir: str | list[str],
        part: str = "train",
        val_size: float = 0.1,
        *args,
        **kwargs
    ):
        data_dir = [data_dir] if isinstance(data_dir, str) else data_dir
        self.data_dir = [ROOT_PATH / dir for dir in data_dir]
        self.part = part
        self.val_size = val_size
        self.formats = (".jpg", ".jpeg", ".png")

        index = []
        for dir in self.data_dir:
            index.extend(self._load_or_create_index(dir))

        super().__init__(index, *args, **kwargs)

    def _load_or_create_index(self, data_dir: Path):
        index = None
        filename = "index_" + self.part + f"_{self.val_size}.json"
        if (data_dir / filename).exists():
            index = read_json(data_dir / filename)
        else:
            index = self._create_index(data_dir)
            write_json(index, data_dir / filename)
        return index

    def _create_index(self, data_dir: Path):
        index = []
        print(f"Creating {data_dir} index, part: {self.part}")

        # fake videos
        for fake_type in (data_dir / "fake").iterdir():
            fake_videos = []
            for frames_path in fake_type.iterdir():
                fake_videos.extend([
                    {
                    "frame_path": str(frame_path),
                    "target": 1
                    }
                    for frame_path in frames_path.iterdir()
                ])
        
            val_size = int(len(fake_videos) * self.val_size)
            if self.part == "train":
                index.extend(fake_videos[:-val_size])
            else:
                index.extend(fake_videos[-val_size:])

        # real videos
        for real_type in (data_dir / "real").iterdir():
            real_videos = []
            for frames_path in real_type.iterdir():
                real_videos.extend([
                    {
                    "frame_path": str(frame_path),
                    "target": 0
                    }
                    for frame_path in frames_path.iterdir()
                ])
        
            val_size = int(len(real_videos) * self.val_size)
            if self.part == "train":
                index.extend(real_videos[:-val_size])
            else:
                index.extend(real_videos[-val_size:])

        return index
    
    def __getitem__(self, idx):
        instance_data = dict(self._index[idx])
        instance_data["frames"] = Image.open(instance_data["frame_path"])
        instance_data["target"] = torch.tensor(instance_data["target"])

        instance_data = self.apply_instance_transforms(instance_data)
        instance_data = self.apply_augmentations(instance_data)

        return instance_data
  