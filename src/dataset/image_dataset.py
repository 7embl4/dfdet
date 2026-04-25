import torch
from PIL import Image

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
        for fake_type in (self.data_dir / "manipulated_sequences").iterdir():
            fake_videos = []
            for fake_dir in (self.data_dir / f"manipulated_sequences/{fake_type.stem}/c23/videos").iterdir():
                fake_videos.append([
                    {
                        "frame_path": str(frame_path),
                        "depth_path": str(frame_path).replace("videos", "depth"),
                        "target": 1
                    } for frame_path in fake_dir.iterdir()
                ])
        
            val_size = int(len(fake_videos) * self.val_size)
            if self.part == "train":
                index.extend(fake_videos[:-val_size])
            else:
                index.extend(fake_videos[-val_size:])

        # real videos
        real_videos = []
        for real_dir in (self.data_dir / "original_sequences/youtube/c23/videos").iterdir():
            real_videos.append([
                {
                    "frame_path": str(frame_path),
                    "depth_path": str(frame_path).replace("videos", "depth"),
                    "target": 0
                } for frame_path in real_dir.iterdir()
            ])
        
        val_size = int(len(real_videos) * self.val_size)
        if self.part == "train":
            index.extend(real_videos[:-val_size])
        else:
            index.extend(real_videos[-val_size:])

        temp = []
        for video in index:
            temp.extend(video)
        index = temp

        return index
    
    def __getitem__(self, idx):
        instance_data = self._index[idx]
        instance_data["rgb"] = Image.open(instance_data["frame_path"])
        instance_data["depth"] = Image.open(instance_data["depth_path"])

        instance_data = self.apply_instance_transforms(instance_data)
        instance_data["rgbd"] = torch.concat(
            (instance_data["rgb"], instance_data["depth"][0].unsqueeze(0)), dim=0
        )

        instance_data = self.apply_augmentations(instance_data)
        return {
            "frame_path": instance_data["frame_path"],
            "depth_path": instance_data["depth_path"],
            "frame": instance_data["rgbd"],
            "target": torch.tensor(instance_data["target"])
        }
