"""VideoMAE with a 10-class head and the k-block unfreezing strategies."""
import torch
from transformers import VideoMAEForVideoClassification


def build_model(cfg):
    classes = cfg["classes"]
    return VideoMAEForVideoClassification.from_pretrained(
        cfg["checkpoint"],
        id2label=dict(enumerate(classes)),  # also sets num_labels
        label2id={c: i for i, c in enumerate(classes)},
        ignore_mismatched_sizes=True,  # drops the Kinetics Linear(768, 400) -> new Linear(768, 10)
    )


def apply_strategy(model, k):
    """Train only the classifier + the last k encoder blocks; everything else stays frozen.

    The pretrained pooling LayerNorm (fc_norm) stays frozen for every k, so the set of trainable
    parameters differs between strategies only by the k blocks.
    """
    blocks = model.videomae.encoder.layer
    if not 0 <= k <= len(blocks):
        raise ValueError(f"k must be in [0, {len(blocks)}], got {k}")
    model.requires_grad_(False)
    model.classifier.requires_grad_(True)
    for block in blocks[len(blocks) - k:]:  # NB: blocks[-k:] would select ALL blocks for k=0
        block.requires_grad_(True)
    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def param_groups(model, cfg):
    """AdamW groups: head vs. backbone learning rate, no weight decay on biases/LayerNorm."""
    groups = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            groups.setdefault((name.startswith("classifier."), p.ndim > 1), []).append(p)
    return [{"params": ps,
             "lr": cfg["lr_head"] if head else cfg["lr_backbone"],
             "weight_decay": cfg["weight_decay"] if decay else 0.0}
            for (head, decay), ps in groups.items()]


def trainable_state(model):
    """Only the trained weights (head + unfrozen blocks); the rest is the public checkpoint."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable(model, path):
    state = torch.load(path, map_location="cpu")
    unexpected = model.load_state_dict(state, strict=False).unexpected_keys
    if unexpected:
        raise KeyError(f"Weights do not match the model: {unexpected[:3]}")
    return model
