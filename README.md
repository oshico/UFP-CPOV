# Human Action Recognition with VideoMAE — UCF101-10

VCOP 2026/27 (UFP). Effect of unfreezing the last *k* ∈ {0, 1, 2} Transformer blocks of
`MCG-NJU/videomae-base-finetuned-kinetics` on accuracy and compute cost.
**Current state: Milestone 1** (pipeline + strategy k=0).

```
pyproject.toml         dependencies (uv.lock pins exact versions); mise.toml: Python, uv, tasks
configs/base.yaml      all experiment settings (fixed by the assignment + group decisions)
splits/                train.csv, validation.csv, test.csv provided by the teacher
vcop/data.py           split loading, leakage check, decoding, temporal sampling, preprocessing
vcop/model.py          VideoMAE + Linear(768, 10), k-block unfreezing, optimiser groups
vcop/utils.py          config overrides, seeding, environment info
train.py               train on train, model selection on validation (test is never read)
infer.py               top-3 predictions on sample videos (Kinetics head or trained head)
scripts/get_ucf101.sh  download UCF101 and extract the 10 classes
```

## Setup

**Local (mise + uv).** `mise.toml` pins Python 3.12 (same as Colab) and uv; `uv.lock` pins every
dependency.
```bash
mise trust && mise install   # Python + uv; .venv is created and auto-activated in the repo
mise run setup               # uv sync --locked
mise run data                # download UCF101, extract the 10 classes into data/
```
On Linux, PyPI's torch wheels include CUDA. On a machine without an NVIDIA GPU the code runs on
CPU, which is fine for the sanity check and smoke tests but not for the real runs.

**Colab (GPU runtime).** Use the preinstalled torch/transformers instead of syncing a second
copy (~3 GB); the versions actually used are logged in every `metrics.json`.
```bash
!git clone https://github.com/<group>/<repo>.git && cd <repo>
!pip install -q av
!bash scripts/get_ucf101.sh data
```
Keep the videos on the local Colab disk (`/content`), not read directly from Drive, which is
much slower. To skip re-downloading, archive `data/UCF-101` to Drive once and copy it back.

## Run

```bash
python infer.py --split validation --n 8              # 1. sanity check   (mise run sanity)
python train.py --set k=0 debug_per_class=4 epochs=2  # 2. smoke test     (mise run train k=0 debug_per_class=4 epochs=2)
python train.py --set k=0                             # 3. strategy A     (mise run train k=0)
python infer.py --split validation --n 8 --weights runs/k0/best.pt
```
Any config key can be overridden with `--set key=value`. Each run writes to `runs/k<k>/`:
`best.pt` (only the trained weights: head + unfrozen blocks, a few KB for k=0) and
`metrics.json` (config, library versions, GPU, total/trainable parameters, epochs run, best
epoch, training time, peak GPU memory, per-epoch history, validation per-class report and
confusion matrix).

Step 1 checks the preprocessing before any training: with the original Kinetics-400 head,
classes that also exist in Kinetics (playing guitar, playing violin, brushing teeth, push up,
typing) should be predicted correctly. If they are not, the pipeline is wrong.

## Experimental protocol (preliminary)

| | Choice | Reason |
|---|---|---|
| Strategies | k=0: head only; k=1/2: head + last k blocks | Everything else frozen, including the pretrained pooling LayerNorm (`fc_norm`), so runs differ only by the k blocks |
| Temporal sampling | `dense`: 16 frames, stride 4, random start (train) / centre clip (eval) | Same temporal scale as Kinetics training of the checkpoint (tubelets pair consecutive sampled frames); `uniform` (whole video) is the alternative to compare on validation |
| Preprocessing | Eval: resize short side 224, centre crop 224, checkpoint mean/std | Identical to the checkpoint's `VideoMAEImageProcessor`, whose stats are read from the Hub |
| Augmentation | RandomResizedCrop (scale 0.7–1), horizontal flip, same on all frames | Mild crops because Lipstick/EyeMakeup/BrushingTeeth differ by small face regions; no class depends on left/right |
| Optimiser | AdamW, lr 1e-3 (head) / 5e-5 (blocks), wd 0.05 (not on bias/LayerNorm) | One configuration for all k, so k remains the only variable |
| Schedule | 1 epoch linear warm-up + cosine, max 20 epochs, grad clip 1.0 | |
| Batch | 8, no accumulation | Fits a T4 in fp16 for k ≤ 2; Colab's 2 CPUs (video decoding) are the bottleneck |
| Selection | Best val accuracy (ties: val loss), early stopping patience 5 | Accuracy is the main metric |
| Precision | fp16 + GradScaler on T4, bf16 on Ampere+ | Recommended mixed precision |
| Seed | 42 | GPU kernels are not fully deterministic |

These values are a starting point and may only be changed based on **validation** results.

## Methodological safeguards

- `test.csv` is only read for its file names, to assert that no UCF101 group (same actor and
  scene) appears in more than one split. Its videos are not loaded until the final evaluation.
- Class labels come from the UCF101 file names and the fixed class order in the config, so
  label ids are identical in every split and run.
- `blocks[len(blocks) - k:]` is used instead of `blocks[-k:]`, which would unfreeze the whole
  backbone when k=0.
- Frozen layers before the first trainable one build no autograd graph, so k=0 stores no
  backbone activations. Training cost therefore grows with k, as intended for the comparison.
