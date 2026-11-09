"""Top-3 predictions on a few dataset videos.

Without --weights the original Kinetics-400 head is used: a pipeline sanity check (e.g. a
PlayingGuitar clip should map to "playing guitar"; wrong preprocessing gives nonsense).
With --weights the trained 10-class head (+ unfrozen blocks) from train.py is used.

    python infer.py --split validation --n 8
    python infer.py --split validation --n 8 --weights runs/k0/best.pt
"""
import argparse
from pathlib import Path

import torch
from transformers import VideoMAEForVideoClassification

from vcop.data import ClipDataset, load_split
from vcop.model import build_model, load_trainable
from vcop.utils import load_config


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--split", default="validation", choices=["train", "validation"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--weights", help="best.pt from train.py; omit to use the Kinetics head")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.weights:
        model = load_trainable(build_model(cfg), args.weights)
    else:
        model = VideoMAEForVideoClassification.from_pretrained(cfg["checkpoint"])
    model.eval().to(device)

    df = load_split(cfg, args.split)
    df = df.sample(min(args.n, len(df)), random_state=cfg["seed"])
    ds = ClipDataset(df, cfg, train=False)
    for i, (path, true) in enumerate(zip(df.path, df.cls)):
        x = ds[i][0].unsqueeze(0).to(device)
        probs = model(pixel_values=x).logits.softmax(-1)[0]
        top = probs.topk(3)
        preds = ", ".join(f"{model.config.id2label[j.item()]} ({p:.2f})"
                          for p, j in zip(top.values, top.indices))
        print(f"{true:<16} -> {preds}   [{Path(path).name}]")


if __name__ == "__main__":
    main()
