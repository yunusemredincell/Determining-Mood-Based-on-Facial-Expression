import os
import random
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FacialExpressionDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

        self.classes = sorted(
            d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
        )
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.samples = []
        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)

            for fname in os.listdir(cls_folder):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                    path = os.path.join(cls_folder, fname)
                    label = self.class_to_idx[cls_name]
                    self.samples.append((path, label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No image files found in {root_dir}. Check the directory structure."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, label


def get_data_transforms(image_size=48):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return transform
