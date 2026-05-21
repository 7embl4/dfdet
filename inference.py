import torch
import torchvision.transforms.v2 as T
import time
from tqdm import tqdm

from src.model import DepthExpert, FAUExpert
from src.dataset import ImageDataset, VideoDataset
from src.metrics import Accuracy, F1, AUC
from src.utils.init import set_random_seed


device = "cuda" if torch.cuda.is_available() else "cpu"
device_tensors = ["frames", "target"]


def move_batch_to_device(batch: dict):
    for t in device_tensors:
        batch[t] = batch[t].to(device)
    return batch


def main():
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
    dataset = VideoDataset(
        data_dir="data/hwei_part1",
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
    model = FAUExpert(encoder_model="ViT-L/14")
    state_dict = torch.load("saved/fau_vcdf_L14/model_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(state_dict=state_dict["state_dict"])
    model.eval()
    model.to(device)

    # inference
    for batch in tqdm(dataloader, desc="test"):
        batch = move_batch_to_device(batch)

        t1 = time.perf_counter()
        with torch.no_grad():
            output = model(**batch)
        t2 = time.perf_counter()
        print(t2 - t1)
        batch.update(output)

        if batch["target"][0].item() != torch.argmax(batch["pred"]).item():
            print(batch["video_path"])

        for metric in metrics:
            metric(**batch)

    # print results
    for metric in metrics:
        print(f"    test_{metric.name:20s}: {metric.avg()}")


if __name__ == "__main__":
    main()
