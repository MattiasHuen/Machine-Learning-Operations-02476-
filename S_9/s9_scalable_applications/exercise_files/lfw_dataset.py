"""LFW dataloading."""

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image
from pathlib import Path


class LFWDataset(Dataset):
    """Initialize LFW dataset."""

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, path_to_folder: str, transform=None, return_path: bool = True) -> None:
        self.root = Path(path_to_folder)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"Dataset root is not a directory: {self.root}")

        self.transform = transform
        self.return_path = return_path

        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: list[tuple[Path, int]] = []
        for c in self.classes:
            class_dir = self.root / c
            for p in class_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in self.IMG_EXTS:
                    self.samples.append((p, self.class_to_idx[c]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under: {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, y = self.samples[index]
        x = Image.open(path).convert("RGB")
        if self.transform is not None:
            x = self.transform(x)

        if self.return_path:
            return x, y, str(path)
        return x, y

def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid((images.detach()[:nmax]), nrow=8).permute(1, 2, 0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-path_to_folder", default="jessicali9530/lfw-dataset/versions/4/lfw-deepfunneled/lfw-deepfunneled", type=str)
    parser.add_argument("-batch_size", default=512, type=int)
    parser.add_argument("-num_workers", default=5, type=int)
    parser.add_argument("-visualize_batch", action="store_true")
    parser.add_argument("-get_timing", action="store_true")
    parser.add_argument("-batches_to_check", default=100, type=int)

    args = parser.parse_args()

    lfw_trans = transforms.Compose([transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)

    # Define dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory = True)

    if args.visualize_batch:
        # Visualize a batch
        batch = next(iter(dataloader))
        images, labels, paths = batch
        images_vis = images.clamp(0, 1) 
        show_images(images_vis, nmax=min(64, images_vis.size(0)))
        plt.tight_layout()
        plt.show()

    if args.get_timing:
        # lets do some repetitions
        mean_time = []
        std_time = []
        workers_count = []
        for workers in range(0, args.num_workers + 1):
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=workers, pin_memory =True)
            res = []
            for _ in range(5):
                start = time.time()
                for batch_idx, _batch in enumerate(dataloader):
                    if batch_idx > args.batches_to_check:
                        break
                end = time.time()

                res.append(end - start)

            res = np.array(res)
            mean_time.append(np.mean(res))
            std_time.append(np.std(res))
            workers_count.append(workers)
            print(f"Timing: {np.mean(res)}+-{np.std(res)}")
        fig, ax = plt.subplots()
        ax.errorbar(workers_count, mean_time, xerr=0.02, yerr=std_time)
        plt.ylabel('Time (s)')
        plt.xlabel('Number of Workers')
        plt.title('Time vs Number of Workers with Uncertainty')
        plt.show()

