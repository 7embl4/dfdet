import torch
import torchvision.transforms.v2 as T

import os
import time
from argparse import ArgumentParser
from tqdm import tqdm

from src.model import DepthExpert, FAUExpert
from src.dataset import ImageDataset, VideoDataset
from src.metrics import Accuracy, F1, AUC
from src.utils.init import set_random_seed
from src.utils.io import write_json


device = "cuda" if torch.cuda.is_available() else "cpu"
device_tensors = ["frames", "target"]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to data"
    )
    parser.add_argument(
        "--data_type", 
        type=str,
        default="video",
        choices=["video", "image"],
        help="Data type to validate"
    )
    parser.add_argument(
        "--save_mistakes", 
        action="store_true",
        help="Saving classification mistakes"
    )

    return parser.parse_args()


def move_batch_to_device(batch: dict):
    for t in device_tensors:
        batch[t] = batch[t].to(device)
    return batch


def main(args):
    set_random_seed(0)

    # dataset
    instance_transforms = {
        "frames": T.Compose([
            T.PILToTensor(),
            T.Resize([224, 224]),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    }

    dataset_type = VideoDataset if args.data_type == "video" else ImageDataset
    dataset = dataset_type(
        data_dir=args.data_path,
        part="val",
        val_size=1.0,
        instance_transforms=instance_transforms
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        drop_last=False,
        shuffle=False
    )

    # metrics
    metrics = [Accuracy(name="accuracy"), F1(name="f1"), AUC(name="auc")]
    
    # model
    model_type = FAUExpert if args.data_type == "video" else DepthExpert
    model = model_type(encoder_model="ViT-L/14")
    folder = "fau" if args.data_type == "video" else "depth"
    state_dict = torch.load(f"models/{folder}/model_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(state_dict=state_dict["state_dict"])
    model.eval()
    model.to(device)

    # inference
    total_time = 0
    mistakes = []
    for batch in tqdm(dataloader, desc="test"):
        batch = move_batch_to_device(batch)

        t1 = time.perf_counter()
        with torch.no_grad():
            output = model(**batch)
        t2 = time.perf_counter()
        total_time += t2 - t1
        batch.update(output)

        target = batch["target"][0].cpu().item()
        pred = torch.argmax(batch["pred"]).cpu().item()
        if pred != target:
            type = "video" if args.data_type == "video" else "frame"
            mistakes.append({
                "path": batch[f"{type}_path"],
                "pred": torch.argmax(batch["pred"]).item(),
                "target": batch["target"][0].item(),
            })

        for metric in metrics:
            metric(**batch)

    # print results
    print(f"    test_{'avg_time':20s}: {total_time / len(dataloader)}")
    for metric in metrics:
        print(f"    test_{metric.name:20s}: {metric.avg()}")

    if args.save_mistakes:
        os.makedirs("saved", exist_ok=True)
        write_json(mistakes, "saved/mistakes.json")

if __name__ == "__main__":
    args = parse_args()
    main(args)
