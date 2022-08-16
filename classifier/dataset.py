# reference: https://github.com/aladdinpersson/Machine-Learning-Collection

import os
import numpy as np
import pandas as pd

from PIL import Image
import torch
from torch.utils.data import Dataset


class CatDogMiniDataset(Dataset):
    def __init__(self, image_dir, transform=None):        
        self.image_dir = image_dir
        self.target = pd.read_csv(image_dir + 'annotations.csv')
        self.transform = transform

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.target.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
        target_label = torch.tensor(int(self.target.iloc[index, 1]))

        if self.transform:
            image = self.transform(image=image)["image"]

        return (image, target_label)


if __name__ == "__main__":

    # quick check
    image_dir = 'data/train/'
    dataset = CatDogMiniDataset(image_dir)
    x = dataset[0]
    print(x)
