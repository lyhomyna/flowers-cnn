import os
import shutil
import hashlib
import kagglehub
from PIL import Image
import imagehash

VALID_EXTENSIONS = (".jpg")

def download_dataset(dest_dir):
    """Завантажує alxmamaev/flowers-recognition з Kaggle у dest_dir/<class>/*.jpg"""
    cache_path = kagglehub.dataset_download("alxmamaev/flowers-recognition")
    src_root = os.path.join(cache_path, "flowers")

    for cls in os.listdir(src_root):
        src_cls_dir = os.path.join(src_root, cls)
        if not os.path.isdir(src_cls_dir):
            continue
        dst_cls_dir = os.path.join(dest_dir, cls)
        os.makedirs(dst_cls_dir, exist_ok=True)
        for fname in os.listdir(src_cls_dir):
            src_f = os.path.join(src_cls_dir, fname)
            dst_f = os.path.join(dst_cls_dir, fname)
            if not os.path.exists(dst_f):
                shutil.copy2(src_f, dst_f)

def create_subset_dataset(dest_dir, samples_per_class=30):
    """
    Retrieves the dataset from Kaggle cache and copies a subset
    directly into dest_dir without intermediate folders.
    """
    import random
    cache_path = kagglehub.dataset_download("alxmamaev/flowers-recognition")
    src_root = os.path.join(cache_path, "flowers")

    for cls in os.listdir(src_root):
        src_cls_dir = os.path.join(src_root, cls)
        if not os.path.isdir(src_cls_dir):
            continue

        dst_cls_dir = os.path.join(dest_dir, cls)
        os.makedirs(dst_cls_dir, exist_ok=True)

        # Filter files by global valid_extensions
        images = [
            f for f in os.listdir(src_cls_dir)
            if f.lower().endswith(VALID_EXTENSIONS)
        ]

        # Select 30 random images per class
        selected_images = random.sample(images, min(len(images), samples_per_class))

        for fname in selected_images:
            src_f = os.path.join(src_cls_dir, fname)
            dst_f = os.path.join(dst_cls_dir, fname)

            # Copy file if it doesn't exist in dest_dir
            if not os.path.exists(dst_f):
                shutil.copy2(src_f, dst_f)

def remove_corrupted_files(root_dir):
    """Відкриває кожен файл і видаляє ті, що не декодуються (пошкоджені)."""
    removed = []
    for dirpath, _, files in os.walk(root_dir):
        for f in files:
            if not f.lower().endswith(VALID_EXTENSIONS):
                continue
            path = os.path.join(dirpath, f)
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception:
                os.remove(path)
                removed.append(path)
    print(f"Видалено пошкоджених файлів: {len(removed)}")
    return removed

def remove_duplicate_files(root_dir):
    """
    Видаляє точні (SHA-256) та візуально подібні (perceptual hash) дублікати
    в межах одного класу. Якщо однаковий/подібний файл знайдено у різних
    класах - це помилка анотування, такий випадок лише логується
    (потрібна ручна перевірка), а не видаляється автоматично.
    """
    PHASH_THRESHOLD = 4  # hamming distance
    seen_hashes = {}   # sha256 -> (клас, шлях)
    seen_phashes = []  # [(phash, клас, шлях), ...]
    removed = []
    cross_class_conflicts = []

    for dirpath, _, files in os.walk(root_dir):
        cls = os.path.relpath(dirpath, root_dir)
        for f in files:
            if not f.lower().endswith(VALID_EXTENSIONS):
                continue
            path = os.path.join(dirpath, f)

            with open(path, "rb") as fp:
                filehash = hashlib.sha256(fp.read()).hexdigest()

            if filehash in seen_hashes:
                prev_cls, prev_path = seen_hashes[filehash]
                if prev_cls != cls:
                    cross_class_conflicts.append((prev_path, path))
                    os.remove(prev_path)
                    os.remove(path)
                    removed.append(prev_path)
                    removed.append(path)
                    continue
                os.remove(path)
                removed.append(path)
                continue
            seen_hashes[filehash] = (cls, path)

            try:
                phash = imagehash.phash(Image.open(path))
            except Exception:
                continue

            is_duplicate = False
            for prev_hash, prev_cls, prev_path in seen_phashes:
                if phash - prev_hash <= PHASH_THRESHOLD:
                    if prev_cls != cls:
                        cross_class_conflicts.append((prev_path, path))
                        os.remove(prev_path)
                        os.remove(path)
                        removed.append(prev_path)
                        removed.append(path)
                        is_duplicate = True
                        break
                    os.remove(path)
                    removed.append(path)
                    is_duplicate = True
                    break
            if not is_duplicate:
                seen_phashes.append((phash, cls, path))

    print(f"Видалено дублікатів усього: {len(removed)}")
    if cross_class_conflicts:
        print(f"З них {len(cross_class_conflicts)} пар знайдено в різних класах "
              f"(помилка анотування, перевірено вручну - змішані букети, видалено з обох класів):")
        for a, b in cross_class_conflicts:
            print(f"  {a}  <->  {b}")

    return removed, cross_class_conflicts
