"""Train strategy k on train, select the best epoch on validation. The test split is never read.

    python train.py --config configs/base.yaml --set k=0
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from vcop.data import SPLITS, ClipDataset, check_no_group_leakage, load_split
from vcop.model import (apply_strategy, build_model, count_parameters, load_trainable,
                        param_groups, trainable_state)
from vcop.utils import env_info, load_config, seed_everything


def amp_setup(device):
    """bf16 on Ampere+ (A100/L4), fp16 + loss scaling on older GPUs (T4)."""
    if device.type != "cuda":
        return torch.float32, False
    return (torch.bfloat16, False) if torch.cuda.get_device_capability()[0] >= 8 \
        else (torch.float16, True)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    total_loss, labels, preds = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
            logits = model(pixel_values=x).logits
        total_loss += F.cross_entropy(logits.float(), y, reduction="sum").item()
        labels.append(y.cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
    labels, preds = np.concatenate(labels), np.concatenate(preds)
    return {"loss": total_loss / len(labels), "acc": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"), "y": labels, "p": preds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. k=1 epochs=2")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg["seed"])
    out = Path(cfg["output_dir"]) / f"k{cfg['k']}"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype, use_scaler = amp_setup(device)

    # ---- data: the leakage check reads only the test file names, never its videos ----
    splits = {s: load_split(cfg, s) for s in SPLITS}
    check_no_group_leakage(splits)
    loader_kw = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
                     pin_memory=device.type == "cuda", persistent_workers=cfg["num_workers"] > 0)
    train_loader = DataLoader(ClipDataset(splits["train"], cfg, train=True), shuffle=True,
                              generator=torch.Generator().manual_seed(cfg["seed"]), **loader_kw)
    val_loader = DataLoader(ClipDataset(splits["validation"], cfg, train=False), **loader_kw)

    # ---- model / optimisation ----
    model = apply_strategy(build_model(cfg), cfg["k"]).to(device)
    params = count_parameters(model)
    print(f"k={cfg['k']} | trainable {params['trainable']:,} / {params['total']:,} | "
          f"train {len(splits['train'])} val {len(splits['validation'])} videos | {amp_dtype}")
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(param_groups(model, cfg))
    steps_per_epoch = math.ceil(len(train_loader) / cfg["grad_accum"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, cfg["warmup_epochs"] * steps_per_epoch, cfg["epochs"] * steps_per_epoch)
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    # ---- training ----
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history, best, best_epoch, bad, train_time = [], None, 0, 0, 0.0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()  # no dropout in this checkpoint; the frozen prefix builds no autograd graph
        t0, run_loss, run_correct, seen = time.perf_counter(), 0.0, 0, 0
        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                logits = model(pixel_values=x).logits
            loss = F.cross_entropy(logits.float(), y)
            scaler.scale(loss / cfg["grad_accum"]).backward()
            if step % cfg["grad_accum"] == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            run_loss += loss.item() * len(y)
            run_correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_time += time.perf_counter() - t0  # training only, validation excluded

        val = evaluate(model, val_loader, device, amp_dtype)
        history.append({"epoch": epoch, "train_loss": run_loss / seen,
                        "train_acc": run_correct / seen, "val_loss": val["loss"],
                        "val_acc": val["acc"], "val_macro_f1": val["macro_f1"]})
        print(" | ".join(f"{name} {v:.4f}" if isinstance(v, float) else f"{name} {v}"
                         for name, v in history[-1].items()))

        score = (val["acc"], -val["loss"])  # main metric, ties broken by validation loss
        if best is None or score > best:
            best, best_epoch, bad = score, epoch, 0
            torch.save(trainable_state(model), out / "best.pt")
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    # ---- report on validation with the selected epoch ----
    peak_mem = round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None
    val = evaluate(load_trainable(model, out / "best.pt"), val_loader, device, amp_dtype)
    labels = list(range(len(cfg["classes"])))
    results = {
        "config": cfg, "env": env_info(), "parameters": params,
        "epochs_run": len(history), "best_epoch": best_epoch,
        "train_time_s": round(train_time, 1), "peak_gpu_mem_gb": peak_mem,
        "val": {"acc": val["acc"], "macro_f1": val["macro_f1"], "loss": val["loss"],
                "report": classification_report(val["y"], val["p"], labels=labels,
                                                target_names=cfg["classes"], output_dict=True,
                                                zero_division=0),
                "confusion_matrix": confusion_matrix(val["y"], val["p"], labels=labels).tolist()},
        "history": history,
    }
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"best epoch {best_epoch}: val acc {val['acc']:.4f}, macro-F1 {val['macro_f1']:.4f} "
          f"| train {train_time / 60:.1f} min | peak GPU {peak_mem} GB -> {out}")


if __name__ == "__main__":
    main()
