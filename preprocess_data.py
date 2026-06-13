"""
Preprocessing data to the format below.

ImageDatasetDirectory
└── real
    └── real_type1
        ├── image1.png
        ├── image2.png
        ├── ...
    ...
└── fake
    └── fake_type1
        ├── image1.png
        ├── image2.png
        ├── ...
    ...

VideoDatasetDirectory
└── real
    └── real_type1
        └── faces
          ├── face1.npz
          ├── face2.npz
          ├── ...
        └── videos
          ├── video1.mp4
          ├── video2.mp4
          ├── ...
        ...
    ...
└── fake
    └── fake_type1
        └── faces
          ├── face1.npz
          ├── face2.npz
          ├── ...
        └── videos
          ├── video1.mp4
          ├── video2.mp4
          ├── ...
        ...
    ...

A `face.npz` is a file with face coordinates for video of format:
[
  [frame_id1, x1, x2, y1, y2],
  [frame_id2, x1, x2, y1, y2],
  ...,
  [frame_idN, x1, x2, y1, y2],
]
"""

import sys
import subprocess

from pathlib import Path
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path", 
        type=str,
        help="Path to data"
    )
    parser.add_argument(
        "--type", 
        type=str,
        default="video",
        choices=["video", "image"],
        help="Type of data (video or image)"
    )
    return parser.parse_args()


def face_extraction(path: Path):
    script_path = Path(__file__).parent / "extract_bboxes.py"
    subprocess.run(
        [
            sys.executable, script_path, 
            "--dataset_path", str(path),
            "--save_path", str(path.parent / "faces")
        ], 
        check=True
    )


def move_files(paths: list[Path], dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    for file in paths:
        file.rename(dest / file.name)


def main(args):
    video_formats = (".mp4", ".avi", ".mov", ".mpeg")
    image_formats = (".jpg", ".jpeg", ".png")
    formats = video_formats if args.type == "video" else image_formats 

    for type in ["real", "fake"]:
        type_path = Path(args.data_path) / type
        paths = [p for p in type_path.iterdir()]
        if all([str(p).endswith(formats) for p in paths]):
            move_files(paths, type_path / type / "videos")
            if args.type == "video":
                face_extraction(type_path / type / "videos")
        elif all([Path(p).is_dir() for p in paths]):
            # real
            #   type1
            #       video1
            #       video2
            #   type2
            #       video1
            #       video2
            for dir in paths:
                dir_paths = [p for p in dir.iterdir()]
                move_files(dir_paths, dir / "videos")
                if args.type == "video":
                    face_extraction(dir / "videos")
        else:
            raise RuntimeError(
                f"""
                Path {type_path} contains directories AND files, 
                please rearrange it in the format of only directories OR only files.
                """
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)