import os
import numpy as np
import pandas as pd
from PIL import Image

"""
Extract From:
    raw/
        - class1/
            - xx.jpg
            - xx.jpg
            - ... 

        - class2/
            - xx.jpg
            - xx.jpg
            - ...

Transform and Load to:
    trian/
        - class1xx.jpg
        - class2xx.jpg
        - class1xx.jpg
        - ...
        - annotation.csv
    val/
        - class1xx.jpg
        - class2xx.jpg
        - class1xx.jpg
        - ...
        - annotation.csv
"""


def main():

    # create destination folders
    train_dir = './train/'
    val_dir = './val/'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_img_list = []
    train_cls_list = []
    val_img_list = []
    val_cls_list = []

    # loop through each class
    raw_dir = './raw/'
    for cls_idx, class_name in enumerate(os.listdir(raw_dir)):

        # loop through each item in a class
        class_dir = os.path.join(raw_dir, class_name)
        for item in os.listdir(class_dir):

            # load an image item to RAM
            try:
                item_path = os.path.join(class_dir, item) 
                image = Image.open(item_path)
                width = image.width
                height = image.height
                if image.mode != 'RGB':
                    image = image.convert('RGB')

            except Exception as e:
                print(e)
                continue  # skip this item if there is an IO error

            # transform the image
            # make it smaller so that it can be easily downloaded for the lecture
            image = image.resize((int(width/4), int(height/4)), resample=Image.BILINEAR)
            
            train_test_ratio = 0.2
            filename = class_name + item
            if np.random.random() > train_test_ratio:
                image.save(os.path.join(train_dir, filename))
                train_img_list.append(filename)
                train_cls_list.append(cls_idx)
            else:
                image.save(os.path.join(val_dir, filename))
                val_img_list.append(filename)
                val_cls_list.append(cls_idx)


    # save annotation files
    df = pd.DataFrame({"img": train_img_list, "cls": train_cls_list})
    df.to_csv(train_dir + 'annotations.csv', index=False)

    df = pd.DataFrame({"img": val_img_list, "cls": val_cls_list})
    df.to_csv(val_dir + 'annotations.csv', index=False)

if __name__ == "__main__":
    main()
