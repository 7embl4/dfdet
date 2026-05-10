import cv2
import random
from pathlib import Path

import torch
import numpy as np

from src.dataset import BaseDataset
from src.utils.io import ROOT_PATH, write_json, read_json


class VideoDataset(BaseDataset):
    def __init__(
        self, 
        # data
        data_dir: str | list[str],
        part: str = "train",
        val_size: float = 0.1,
        chunk_size: int = 32,
        max_res: int = 640,
        output_size: int = 224,

        # other
        *args,
        **kwargs
    ):
        data_dir = [data_dir] if isinstance(data_dir, str) else data_dir
        self.data_dir = [ROOT_PATH / dir for dir in data_dir]
        self.part = part
        self.val_size = val_size
        self.chunk_size = chunk_size
        self.max_res = max_res
        self.output_size = output_size
        self.formats = (".mp4", ".avi", ".mov", ".mpeg")

        index = []
        for dir in self.data_dir:
            index.extend(self._load_or_create_index(dir))
        
        super().__init__(index, *args, **kwargs)

    def _load_or_create_index(self, data_dir: Path):
        """
        Load existing index or create if it doesn't exist.
        """
        local_index = None
        filename = "index_" + self.part + f"_{self.val_size}.json"
        if (data_dir / filename).exists():
            local_index = read_json(data_dir / filename)
        else:
            local_index = self._create_index(data_dir)
            write_json(local_index, data_dir / filename)
        return local_index

    def _create_index(self, data_dir: Path):
        """
        Creates index file for a dataset.
        """
        index = []
        print(f"Creating {data_dir} index, part: {self.part}")

        # fake videos
        for fake_type in (data_dir / "fake").iterdir():
            fake_videos = []
            for video_path in (fake_type / "videos").iterdir():
                face_path = str(video_path).replace("videos", "faces").replace(video_path.suffix, ".npz")
                fake_videos.append({
                    "video_path": str(video_path),
                    "face_path": face_path,
                    "target": 1
                })
        
            val_size = int(len(fake_videos) * self.val_size)
            if self.part == "train":
                index.extend(fake_videos[:-val_size])
            else:
                index.extend(fake_videos[-val_size:])

        # real videos
        for real_type in (data_dir / "real").iterdir():
            real_videos = []
            for video_path in (real_type / "videos").iterdir():
                face_path = str(video_path).replace("videos", "faces").replace(video_path.suffix, ".npz")
                real_videos.append({
                    "video_path": str(video_path),
                    "face_path": face_path,
                    "target": 0
                })
        
            val_size = int(len(real_videos) * self.val_size)
            if self.part == "train":
                index.extend(real_videos[:-val_size])
            else:
                index.extend(real_videos[-val_size:])
            
        return index
    
    def __getitem__(self, idx):
        """
        Gets video from a dataset and applies transforms on it.
        """
        instance_data = dict(self._index[idx])
        frames = self._read_video_and_get_faces(instance_data)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
        instance_data.update({"frames": frames})

        instance_data = self.apply_instance_transforms(instance_data)
        instance_data = self.apply_augmentations(instance_data)
        instance_data.update({"target": torch.tensor(instance_data["target"])})

        return instance_data

    def _read_video_and_get_faces(self, instance_data: dict):
        """
        Gets random chunk of a video and crop faces from it.
        Randomly generates start of a chunk.

        Args:
            video_path (str): path to video
        """
        cap = cv2.VideoCapture(instance_data["video_path"].strip())
        face_data = np.load(instance_data["face_path"].strip())
        bboxes = face_data["bboxes"]

        # find indices
        indices = []
        current_chunk = [bboxes[0, 0]]
        for i in range(1, len(bboxes)):
            if abs(bboxes[i, 0] - bboxes[i - 1, 0]) > 1:
                if len(current_chunk) >= self.chunk_size:
                    indices.extend(current_chunk[:-self.chunk_size + 1])
                current_chunk.clear()
            current_chunk.append(bboxes[i, 0])

        if len(current_chunk) >= self.chunk_size:
            indices.extend(current_chunk[:-self.chunk_size + 1])

        # set capture
        if self.part != "train":
            random.seed(0)
        bboxes = {ind: bbox for ind, bbox in zip(bboxes[:, 0], bboxes[:, 1:])}
        start = random.choice(indices)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        # lower image resolution if it's too high
        video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        long_side = max(video_width, video_height)
        scale = self.max_res / long_side if long_side > self.max_res else None

        # get faces
        faces = np.empty((self.chunk_size, self.output_size, self.output_size, 3), dtype=np.uint8)
        for i in range(self.chunk_size):
            ret, frame = cap.read()
            if not ret:
                print(
                    f"Warning: cannot read {start + i}th frame of video {instance_data['video_path']}"
                )
                continue
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if scale:
                frame = cv2.resize(frame, (int(video_width * scale), int(video_height * scale)))
            
            x1, x2, y1, y2 = bboxes[start + i]
            face = frame[y1:y2, x1:x2]
            faces[i] = cv2.resize(face, (self.output_size, self.output_size))

        cap.release()
        return faces
