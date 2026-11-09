"""UCF101-10 pipeline: split loading -> decoding -> temporal sampling -> preprocessing."""
import re
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
from transformers import VideoMAEImageProcessor

SPLITS = ("train", "validation", "test")
UCF_NAME = re.compile(r"v_(?P<cls>[A-Za-z]+)_g(?P<group>\d+)_c\d+")


def load_split(cfg, split):
    """Read a split CSV -> DataFrame[path, cls, group, label].

    Class and group are parsed from the UCF101 file name (v_<Class>_gXX_cYY.avi), so only a
    column with the file name/path is needed and labels follow the fixed order in the config.
    """
    df = pd.read_csv(Path(cfg["splits_dir"]) / f"{split}.csv")
    col = cfg["path_column"] or next(
        (c for c in df.columns if all(UCF_NAME.search(str(v)) for v in df[c])), None)
    if col is None:
        raise ValueError(f"{split}.csv: no column of UCF101 video names; set 'path_column'.")

    names = df[col].astype(str).tolist()
    parsed = [UCF_NAME.search(Path(p).name) for p in names]
    if None in parsed:
        raise ValueError(f"{split}.csv: not a UCF101 name: {names[parsed.index(None)]}")
    out = pd.DataFrame({"rel": names, "cls": [m["cls"] for m in parsed],
                        "group": [int(m["group"]) for m in parsed]})
    unknown = set(out.cls) - set(cfg["classes"])
    if unknown:
        raise ValueError(f"{split}.csv has classes outside the config: {sorted(unknown)}")
    out["label"] = out.cls.map({c: i for i, c in enumerate(cfg["classes"])})

    root = Path(cfg["video_root"])
    out["path"] = [_resolve(root, r, c) for r, c in zip(out.rel, out.cls)]
    missing = out.rel[out.path.isna()]
    if len(missing):
        raise FileNotFoundError(f"{len(missing)} videos of {split} not found under {root}, "
                                f"e.g. {missing.iloc[0]}")

    if cfg.get("debug_per_class"):
        out = out.groupby("cls").head(cfg["debug_per_class"])
    return out.drop(columns="rel").reset_index(drop=True)


def _resolve(root, rel, cls):
    rel = rel if rel.lower().endswith(".avi") else rel + ".avi"
    for p in (root / rel, root / cls / Path(rel).name):
        if p.exists():
            return str(p)
    return None


def check_no_group_leakage(splits):
    """UCF101 clips of the same (class, group) share actor/scene: they must not cross splits."""
    keys = {s: set(zip(df.cls, df.group)) for s, df in splits.items()}
    for a, b in combinations(keys, 2):
        if keys[a] & keys[b]:
            raise ValueError(f"Group leakage between {a} and {b}: {sorted(keys[a] & keys[b])[:5]}")


def sample_indices(n, num_frames, strategy, train, stride=4):
    """Pick `num_frames` frame indices from a video with `n` frames."""
    if strategy == "uniform":  # TSN-style: one frame per equal segment, covers the whole video
        edges = np.linspace(0, n, num_frames + 1)
        offset = torch.rand(num_frames).numpy() if train else 0.5  # torch RNG: seeded per worker
        idx = edges[:-1] + offset * np.diff(edges)
    elif strategy == "dense":  # one contiguous clip with fixed stride, as in Kinetics training
        span = (num_frames - 1) * stride + 1
        if n < span:
            idx = np.linspace(0, n - 1, num_frames)
        else:
            start = int(torch.randint(n - span + 1, ())) if train else (n - span) // 2
            idx = start + stride * np.arange(num_frames)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    return np.clip(idx.astype(int), 0, n - 1)


def read_frames(path, pick):
    """Decode a video once and convert only the selected frames -> uint8 (T, H, W, 3).

    Frame counts in AVI headers are unreliable and seeking is imprecise, so the video is decoded
    sequentially (UCF101 clips are short: ~7 s at 320x240).
    """
    with av.open(path) as container:
        frames = list(container.decode(video=0))
    return np.stack([frames[i].to_ndarray(format="rgb24") for i in pick(len(frames))])


@lru_cache
def norm_stats(checkpoint):
    """Normalisation constants of the checkpoint's own preprocessor (single source of truth)."""
    proc = VideoMAEImageProcessor.from_pretrained(checkpoint)
    return list(proc.image_mean), list(proc.image_std)


def build_transform(cfg, train):
    size = cfg["image_size"]
    if train:
        spatial = [v2.RandomResizedCrop(size, scale=tuple(cfg["crop_scale"]), antialias=True)]
        spatial += [v2.RandomHorizontalFlip()] if cfg["hflip"] else []
    else:
        spatial = [v2.Resize(size, antialias=True), v2.CenterCrop(size)]  # = HF processor
    mean, std = norm_stats(cfg["checkpoint"])
    # v2 transforms draw random parameters once per call -> identical crop/flip on all frames
    return v2.Compose([*spatial, v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)])


class ClipDataset(Dataset):
    """Returns (pixel_values (T, C, H, W) float32, label)."""

    def __init__(self, df, cfg, train):
        self.paths, self.labels = df.path.tolist(), df.label.tolist()
        self.transform = build_transform(cfg, train)
        self.sampling = dict(num_frames=cfg["num_frames"], strategy=cfg["sampling"],
                             stride=cfg["dense_stride"], train=train)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        frames = read_frames(self.paths[i], lambda n: sample_indices(n, **self.sampling))
        video = tv_tensors.Video(torch.from_numpy(frames).permute(0, 3, 1, 2))
        return self.transform(video).as_subclass(torch.Tensor), self.labels[i]
