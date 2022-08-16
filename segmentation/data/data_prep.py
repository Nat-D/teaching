"""
Extract from
    raw/
        - data_semantics
            -training
                -image_2
                -instance
                -semantic
                -semantic_rgb
            -testing
                -image_2

Transform and Load to
    train/
        - image
        - mask
    val/
        - image
        - mask

"""
import os 
import numpy as np
from PIL import Image


color_group = np.array([[0,0,0],  # black - other
                       [255,0,0], # red - road 
                       [0, 255, 0],  # green - vehicle
                       ],dtype=np.uint8) 

def main():
    
    # create destination folders
    train_dir = './train/'
    val_dir = './val/'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for folder in ['image', 'mask', 'rgb']:
        os.makedirs(train_dir+folder, exist_ok=True)
        os.makedirs(val_dir+folder, exist_ok=True)


    # looop through each item
    source_image_dir = './raw/data_semantics/training/image_2/'
    source_mask_dir = './raw/data_semantics/training/semantic/'

    for item in os.listdir(source_image_dir):

        try:
            image_path = os.path.join(source_image_dir, item) 
            image = Image.open(image_path).convert('RGB')

            mask_path = os.path.join(source_mask_dir, item) 
            mask = Image.open(mask_path).convert('L')

        except Exception as e:
            print(e)
            continue

        # transform the image and mask
        width = 310
        height = 94

        image = image.resize( (width, height), resample=Image.BILINEAR)
        mask  = mask.resize(  (width, height), resample=Image.NEAREST)

        mask = np.array(mask, dtype=np.uint8)
        mask_old = mask.copy()
        mask[mask_old < 7] = 0 # other
        mask[mask_old == 7] = 1 # road
        mask[ 7 < mask_old] = 0 # other
        mask[ 24 < mask_old] = 2 # vehicle (car, truck, train, rider, etc.)
        
        
        rgb_mask = color_group[mask] 

        mask = Image.fromarray(mask)
        rgb_mask = Image.fromarray(rgb_mask)

        train_test_ratio = 0.2
        if np.random.random() > train_test_ratio:
            image.save(os.path.join(train_dir+'image', item))
            mask.save(os.path.join(train_dir+'mask', item))
            rgb_mask.save(os.path.join(train_dir+'rgb', item))
        else:
            image.save(os.path.join(val_dir+'image', item))
            mask.save(os.path.join(val_dir+'mask', item))
            rgb_mask.save(os.path.join(val_dir+'rgb', item))



if __name__ == "__main__":
    main()