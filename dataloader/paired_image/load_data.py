from __future__ import annotations

import os
import random
from glob import glob

import torch.utils.data as data
from PIL import Image, ImageFile
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


def create_dataset(args, data_path: str, mode: str = "train"):
    if args.DATA_TYPE != "PRISM":
        raise ValueError(f"This release supports DATA_TYPE='PRISM' only, got {args.DATA_TYPE!r}.")

    image_files = sorted(glob(os.path.join(data_path, "input", "*.png")))
    dataset = PRISMPairedDataset(args, image_files, mode=mode)

    distributed = bool(getattr(args, "DISTRIBUTED", False))
    sampler = None
    shuffle = mode == "train"
    drop_last = mode == "train"

    if distributed and mode == "train":
        world_size = int(getattr(args, "WORLD_SIZE", 1))
        rank = int(getattr(args, "RANK", 0))
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        shuffle = False

    return data.DataLoader(
        dataset,
        batch_size=args.BATCH_SIZE,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.WORKER,
        drop_last=drop_last,
    )


class PRISMPairedDataset(data.Dataset):
    def __init__(self, args, image_list: list[str], mode: str = "train"):
        self.image_list = image_list
        self.args = args
        self.mode = mode
        self.loader = args.LOADER

    def __getitem__(self, index: int):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        path_src = self.image_list[index]
        base = os.path.basename(path_src)
        number = os.path.splitext(base)[0]
        gt_id = base.split("_")[0]
        split_root = os.path.dirname(os.path.dirname(path_src))
        path_tar = os.path.join(split_root, "gt", f"{gt_id}.png")

        item: dict[str, object] = {}
        if self.mode == "train":
            labels, moire_imgs = self._load_train_pair(path_tar, path_src)
        elif self.mode == "test":
            labels, moire_imgs = self._load_test_pair(path_tar, path_src, item)
        else:
            raise ValueError(f"Unrecognized mode {self.mode!r}; expected 'train' or 'test'.")

        item["in_img"] = moire_imgs
        item["label"] = labels
        item["number"] = number
        return item

    def __len__(self):
        return len(self.image_list)

    def _load_train_pair(self, path_tar: str, path_src: str):
        if self.loader == "crop":
            with Image.open(path_tar) as im:
                w, h = im.size
            crop_size = self.args.CROP_SIZE
            if w < crop_size or h < crop_size:
                return resize_loader(crop_size, [path_tar, path_src])
            x = random.randint(0, w - crop_size)
            y = random.randint(0, h - crop_size)
            return crop_loader(crop_size, x, y, [path_tar, path_src])

        if self.loader == "resize":
            return resize_loader(self.args.RESIZE_SIZE, [path_tar, path_src])

        if self.loader == "default":
            return default_loader([path_tar, path_src])

        raise ValueError(f"Unrecognized LOADER {self.loader!r}.")

    def _load_test_pair(self, path_tar: str, path_src: str, item: dict[str, object]):
        if self.loader == "resize":
            labels, moire_imgs = resize_loader(self.args.RESIZE_SIZE, [path_tar, path_src])
            item["origin_label"] = default_loader([path_tar])[0]
            return labels, moire_imgs

        return default_loader([path_tar, path_src])


def default_loader(path_set: list[str]):
    return [default_to_tensor(Image.open(path).convert("RGB")) for path in path_set]


def crop_loader(crop_size: int, x: int, y: int, path_set: list[str]):
    imgs = []
    for path in path_set:
        img = Image.open(path).convert("RGB")
        img = img.crop((x, y, x + crop_size, y + crop_size))
        imgs.append(default_to_tensor(img))
    return imgs


def resize_loader(resize_size: int, path_set: list[str]):
    imgs = []
    for path in path_set:
        img = Image.open(path).convert("RGB")
        img = img.resize((resize_size, resize_size), Image.BICUBIC)
        imgs.append(default_to_tensor(img))
    return imgs


def default_to_tensor(img: Image.Image):
    return transforms.ToTensor()(img)
