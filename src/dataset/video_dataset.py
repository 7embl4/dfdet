import cv2
import random

import torch
import numpy as np
from PIL import Image

from src.dataset import BaseDataset
from src.utils.io import ROOT_PATH, write_json, read_json
import time

class VideoDataset(BaseDataset):
    def __init__(
        self, 
        # data
        data_dir,
        part="train",
        val_size=0.1,
        chunk_size=32,

        # face detector 
        extractor="models/face_detection_yunet_2023mar.onnx",
        input_size=320,
        output_size=224,
        score_th=0.7,
        nms_th=0.4,
        top_k=5000,
        margin=50,
        num_repeats=16,
        n_frames=8,
        max_res=640,

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
        self.num_repeats = num_repeats
        self.n_frames = n_frames
        self.max_res = max_res

        index = self._load_or_create_index()
        super().__init__(index, *args, **kwargs)

    def _load_or_create_index(self):
        """
        Load existing index or create if it doesn't exist.
        """
        index = None
        filename = "index_" + self.part + f"_{self.val_size}.json"
        if (self.data_dir / filename).exists():
            index = read_json(self.data_dir / filename)
        else:
            index = self._create_index()
            write_json(index, self.data_dir / filename)
        return index

    def _create_index(self):
        """
        Creates index file for a dataset.
        """
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
        """
        Gets video from a dataset and applies transforms on it.
        """
        instance_data = dict(self._index[idx])
        frames = self._read_video_and_get_faces(instance_data["video_path"])
        if frames is None:
            print(f"Warning: No valid frames in {instance_data['video_path']}")
            instance_data.update({"frames": None})
            return instance_data

        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
        instance_data.update({"frames": frames})

        instance_data = self.apply_instance_transforms(instance_data)
        instance_data = self.apply_augmentations(instance_data)
        instance_data.update({"target": torch.tensor(instance_data["target"])})

        return instance_data

    def _read_video_and_get_faces(self, video_path: str):
        """
        Gets random chunk of a video and crop faces from it.
        Randomly generates start of a chunk until obtained chunk is valid (has faces for whole duration)
        or until `self.num_repeats` is reached (which statistically almost impossible).

        Args:
            video_path (str): path to video
        """
        cap = cv2.VideoCapture(video_path.strip())
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        check_indices = np.linspace(0, self.chunk_size - 1, self.n_frames).astype(np.int64)

        faces = []
        for _ in range(self.num_repeats):
            start = random.randint(0, total_frames - self.chunk_size)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)

            indices = []
            frames = []
            for ind in range(self.chunk_size):
                ret, frame = cap.read()
                if not ret:
                    print(f"Warning: cannot read {start + ind}th frame of video {video_path}")
                    continue
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                
                if ind in check_indices:
                    face = self._get_face(frame)                    
                    if face:
                        faces.append(face)
                        indices.append(ind)

            # found face on first and last frames and
            # found face on at least half of frames
            if (
                0 in indices and (self.chunk_size - 1) in indices 
                and len(faces) >= self.n_frames // 2
            ):
                faces = self._interpolate_faces(faces, indices, frames)
                break
            else:
                faces.clear()
                frames.clear()
                indices.clear()

        cap.release()
        return faces if len(faces) > 0 else None

    def _get_face(self, frame: np.ndarray):
        """
        Get face from a video using `YuNet` face detection model.

        Args:
            frame (np.nparray): frame in numpy format (H, W, C)
        """
        # lower image resolution if it's to high,
        # since YuNet can't find faces on high resolutions
        long_side = max(frame.shape[:2])
        if long_side > self.max_res:
            scale = self.max_res / long_side
            frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

        # set up detector and detect faces
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, detected_faces = self.detector.detect(frame)

        # crop the largest face 
        if detected_faces is not None:
            x, y, w, h = detected_faces[0][:4].astype(np.int32)
            x1, x2 = max(x - self.margin, 0), min(x + w + self.margin, width)
            y1, y2 = max(y - self.margin, 0), min(y + h + self.margin, height)
            
            face_crop = frame[y1:y2, x1:x2]
            face_crop = cv2.resize(face_crop, (self.output_size, self.output_size))
        
            return {
                "face_crop": face_crop,
                "bbox": (x1, x2, y1, y2)
            }
        
        return None

    def _interpolate_faces(self, faces: list[dict], indices: list[int], frames: list[np.ndarray]):
        """
        Interpolates faces between found ones by averaging 
        bounding boxes of closest frames with detected faces.

        Args:
            faces (list[dict]): found faces in format of dicts 
                with `face_crop` (image of a face) and `bbox` (coords of a face)
            indices (list[int]): frames indices in chunk (0...chunk_size-1) where faces were found
            frames (list[np.ndarray]): all frames of a chunk
        """
        interpolated_faces = [faces[0]["face_crop"]]
        for i in range(len(indices) - 1):
            left = indices[i] + 1
            right = indices[i + 1]

            # get bboxes and interpolate them
            left_bbox = faces[i]["bbox"]
            right_bbox = faces[i + 1]["bbox"]
            inter_bbox = [
                int((left_coord + right_coord) / 2) 
                for left_coord, right_coord in zip(left_bbox, right_bbox)
            ]
            x1, x2, y1, y2 = inter_bbox

            # take faces from intermediate frames
            for j in range(left, right):
                face = frames[j][y1:y2, x1:x2]
                face = cv2.resize(face, (self.output_size, self.output_size))
                interpolated_faces.append(face)

            interpolated_faces.append(faces[i + 1]["face_crop"])

        return np.stack(interpolated_faces)
