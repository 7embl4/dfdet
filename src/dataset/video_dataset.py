import cv2
import random

import torch
import numpy as np
from PIL import Image

from src.dataset import BaseDataset
from src.utils.io import ROOT_PATH, write_json, read_json


class VideoDataset(BaseDataset):
    def __init__(
        self, 
        # data
        data_dir,
        part,
        val_size,
        chunk_size,

        # face detector 
        extractor,
        input_size,
        output_size,
        score_th,
        nms_th,
        top_k,
        margin,

        # other
        *args,
        **kwargs
    ):
        self.data_dir = ROOT_PATH / data_dir
        self.part = part
        self.val_size = val_size
        self.chunk_size = chunk_size
        self.formats = (".mp4", ".avi", ".mov", ".mpeg")

        self.detector = cv2.FaceDetectorYN.create(
            extractor, "",
            (input_size, input_size),
            score_th,
            nms_th,
            top_k,
        )
        self.output_size = output_size
        self.margin = margin

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
            for video_path in fake_type.iterdir():
                fake_videos.append({
                    "video_path": str(video_path),
                    "target": 1
                })
        
            val_size = int(len(fake_videos) * self.val_size)
            if self.part == "train":
                index.extend(fake_videos[:-val_size])
            else:
                index.extend(fake_videos[-val_size:])

        # real videos
        real_videos = []
        for video_path in (self.data_dir / "real").iterdir():
            real_videos.append({
                "video_path": str(video_path),
                "target": 0
            })
        
        val_size = int(len(real_videos) * self.val_size)
        if self.part == "train":
            index.extend(real_videos[:-val_size])
        else:
            index.extend(real_videos[-val_size:])

        return index
    
    def __getitem__(self, idx):
        instance_data = self._index[idx]
        frames = self._read_video(instance_data["video_path"])
        frames = self._get_faces(frames)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
        instance_data.update({"frames": frames})

        instance_data = self.apply_instance_transforms(instance_data)
        instance_data = self.apply_augmentations(instance_data)
        instance_data.update({"target": torch.tensor(instance_data["target"])})

        return instance_data

    def _read_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path.strip())
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = random.randint(0, total_frames - self.chunk_size)
        end = start + self.chunk_size
        
        frames = []
        for ind in range(start, end):
            cap.set(cv2.CAP_PROP_POS_FRAMES, ind)
            ret, frame = cap.read()
        
            if not ret:
                print(f"Warning: cannot read {ind}th frame of video {video_path}")
                continue
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        return frames

    def _get_faces(self, frames: list[np.ndarray]):
        faces = []
        for frame in frames:
            height, width = frame.shape[:2]
            self.detector.setInputSize((width, height))
            _, detected_faces = self.detector.detect(frame)

            if detected_faces is not None:
                x, y, w, h = detected_faces[0][:4].astype(np.int32)
                x1, x2 = max(x - self.margin, 0), min(x + w + self.margin, width)
                y1, y2 = max(y - self.margin, 0), min(y + h + self.margin, height)
                
                face_crop = frame[y1:y2, x1:x2]
                face_crop = cv2.resize(face_crop, (self.output_size, self.output_size))
                faces.append(face_crop)
        
        return np.stack(faces)
