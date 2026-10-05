import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

IMG_SIZE = (128, 128)

def augment_image(img):
    augmentations = {
        'h_flip': img.transpose(Image.FLIP_LEFT_RIGHT),
        'rotate': img.rotate(random.uniform(-30, 30), expand=True, fillcolor=(255,255,255)),
        'bright': ImageEnhance.Contrast(ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 2.0))).enhance(random.uniform(0.5, 2.0)),
        'noise': Image.fromarray(np.clip(np.array(img) + np.random.normal(0, 25, np.array(img).shape), 0, 255).astype(np.uint8)),
        'jitter': Image.fromarray(np.array(img.convert('HSV').convert('RGB'))),
        'blur': img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1)))
    }
    return augmentations

def process_and_save_image(target_dir, image_path, augmentation):
    image_name = os.path.basename(image_path)
    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    img.save(os.path.join(target_dir, image_name))

    if augmentation == True:
        # augmentation
        for name, aug_img in augment_image(img).items():
            if aug_img.size != IMG_SIZE:
                aug_img = aug_img.resize(IMG_SIZE)
            name_parts = image_name.rsplit('.', 1)
            name_aug = f"{name_parts[0]}_{name}.{name_parts[1]}"
            aug_img.save(os.path.join(target_dir, name_aug))

def save_files_to_dir(files, target_dir, split):
    for f in files:
        if split == "train":
            process_and_save_image(target_dir, f, augmentation=True)
        else:
            process_and_save_image(target_dir, f, augmentation=False)

def split_files(target_files):
    n = len(target_files)
    train_end = int(n * 0.8)
    val_end = train_end + int(n * 0.1)

    return {
        "train": target_files[:train_end],
        "val": target_files[train_end:val_end],
        "test": target_files[val_end:]
    }
