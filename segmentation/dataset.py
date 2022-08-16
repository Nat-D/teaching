# reference: https://github.com/aladdinpersson/Machine-Learning-Collection

import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class KittiSegmentMini(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        image = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask  = augmentations["mask"]

        return image, mask


if __name__ == "__main__":

    from torch.utils.data import DataLoader
    import albumentations as A
    from albumentations.pytorch import ToTensorV2


    train_transform = A.Compose([
                        A.Resize(height=94, width=310),
                        ToTensorV2()
                        ])
 
    train_data_object = KittiSegmentMini(image_dir='data/train/image/',
                                         mask_dir='data/train/mask/')

    train_loader = DataLoader(train_data_object,
                              batch_size=8)

    x,y = next(iter(train_loader))
    print(x)