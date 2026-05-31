import cv2
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    
    # dataset
    parser.add_argument("--dataset_path", type=str)
    
    # extractor
    parser.add_argument("--extractor", type=str, default="face_detection_yunet_2023mar.onnx")
    parser.add_argument("--input_size", type=int, default=320)
    parser.add_argument("--output_size", type=int, default=256)
    parser.add_argument("--margin", type=int, default=50)
    parser.add_argument("--score_th", type=float, default=0.7)
    parser.add_argument("--nms_th", type=float, default=0.4)
    parser.add_argument("--top_k", type=int, default=5000)
    parser.add_argument("--inter_step", type=int, default=4)

    args = parser.parse_args()
    return args


def create_video_list(args):
    formats = (".mp4", ".avi", ".mov", ".mpeg")
    dataset_path = Path(args.dataset_path)
    if (dataset_path / "video_list.txt").exists():
        return
    
    with open(str(dataset_path / "video_list.txt"), "w", encoding="utf-8") as video_list:
        for video_path in dataset_path.rglob("*"):
            if str(video_path).endswith(formats):
                video_list.write(str(video_path) + "\n")


def interpolate_bboxes(bboxes: np.ndarray):
    result = [bboxes[0]]
    for i in range(len(bboxes) - 1):
        left_bbox = bboxes[i]
        right_bbox = bboxes[i + 1]
        for j in range(1, right_bbox[0] - left_bbox[0]):
            inter_bbox = [
                int((left_coord + right_coord) / 2) 
                for left_coord, right_coord in zip(left_bbox[1:], right_bbox[1:])
            ]
            result.append([
                left_bbox[0] + j, *inter_bbox
            ])
        result.append(right_bbox)
    return np.stack(result)


def extract_faces(args):
    detector = cv2.FaceDetectorYN.create(
        args.extractor, "",
        (args.input_size, args.input_size),
        args.score_th,
        args.nms_th,
        args.top_k,
    )

    with open(args.dataset_path + "/video_list.txt", "r", encoding="utf-8") as video_list:
        for video in tqdm(video_list.readlines()):
            # capture video
            video_path = Path(video.strip())  # not using strip causes slowing in frames reading (idk why)
            cap = cv2.VideoCapture(str(video_path).strip())
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            check_indices = set([
                int(ind) for ind in np.linspace(0, total_frames - 1, total_frames // args.inter_step)
            ])

            # scaling
            video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            long_side = max(video_width, video_height)
            scale = 640 / long_side if long_side > 640 else None

            # save path
            save_path = Path(str(video_path.parent).replace("\\", "_bboxes\\"))
            save_path.mkdir(parents=True, exist_ok=True)

            # get frames
            saved = []
            for ind in range(total_frames):
                # grab frame
                grabbed = cap.grab()
                if not grabbed:
                    print(f"Warning: cannot read video {str(video_path)}")
                    break
                
                # skip if not for checking
                if ind not in check_indices:
                    continue

                # read frame
                ret, frame = cap.retrieve()
                if not ret:
                    print(f"Warning: cannot read video {str(video_path)}")
                    break

                # process frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if scale:
                    frame = cv2.resize(frame, (int(video_width * scale), int(video_height * scale)))

                # extract faces
                height, width = frame.shape[:2]
                detector.setInputSize((width, height))
                _, faces = detector.detect(frame)

                # save faces
                if faces is not None:
                    x, y, w, h = faces[0][:4].astype(np.int32)
                    x1, x2 = max(x - args.margin, 0), min(x + w + args.margin, width)
                    y1, y2 = max(y - args.margin, 0), min(y + h + args.margin, height)
                    
                    # saving
                    saved.append([ind, x1, x2, y1, y2])

            if len(saved) == 0:
                print(f"Warning: couldn't find any face frames in {str(video_path)}")
                continue

            saved = np.array(saved)
            saved = interpolate_bboxes(saved)
            with (save_path / f"{video_path.stem}.npz").open("wb") as save_file:
                np.savez(save_file, bboxes=saved)

            cap.release()


def main(args):
    create_video_list(args)
    extract_faces(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
