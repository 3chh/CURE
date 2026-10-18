# Training a backbone

`cure-ood train` trains a feature extractor on the same ID training split as the benchmark (OpenOOD v1.5 imglists), then writes the extraction and benchmark configs so that the new features go through `cure-ood extract` and `cure-ood run` like the shipped ones. The point is to test whether a better feature extractor changes the ranking of CURE and the baselines, so the data, splits and evaluation stay fixed and only the network and its training objective change.

## Quick start

```bash
cure-ood train cifar10_resnet18_ce               # ~1.5 h on an RTX 3050 laptop GPU
cure-ood extract cifar10_resnet18_ce             # writes $CURE_FEATURES/cifar10_resnet18_ce.npz
cure-ood train cifar10_resnet18_ce --select-k    # K of the kNN scorer by APS on (id_val, val_tin)
cure-ood run robustness --benchmark cifar10_resnet18_ce
cure-ood report robustness --benchmark cifar10_resnet18_ce
```

Images and imglists are found exactly as in feature extraction: `$CURE_IMAGES/<image_root>/<relpath>` and `$CURE_IMGLISTS/<imglists>/<list>.txt`. Outputs go to `results/train/<run>/`:

| File | Content |
|---|---|
| `config.yaml` | the resolved config (after `--set`) |
| `log.jsonl` | one line per epoch: stage, loss, lr, seconds, images/s, peak GPU memory, val accuracy |
| `state.pt` | full training state, rewritten after every epoch |
| `best.ckpt`, `last.ckpt` | network `state_dict` only; loads with `build_backbone(<backbone>, path)` |

A run resumes from `state.pt` automatically; `--fresh` starts over. Shuffling and augmentation are seeded per (stage, epoch), so a resumed run follows the same data order as an uninterrupted one (cuDNN kernels are not deterministic, so the losses agree to about 4 digits, not bit for bit).

Options:

```
cure-ood train <config> [--device cuda] [--output DIR] [--fresh] [--set KEY=VALUE ...] [--select-k]
```

`--set` overrides any field with a dotted path; list indices are integers and values are parsed as YAML, e.g. `--set data.subsample=2000 --set stages.0.epochs=1 --set stages.0.optimizer.lr=0.05`. Change `run` as well when the override changes the experiment, otherwise the run directory and the generated configs are shared.

## Shipped configs

| Config | Recipe | Source |
|---|---|---|
| `cifar10_resnet18_ce`, `cifar100_resnet18_ce` | ResNet-18 32x32, cross-entropy, SGD (Nesterov) lr 0.1, momentum 0.9, wd 5e-4, batch 128, 100 epochs, per-step cosine to 1e-6; augmentation Resize 32, CenterCrop 32, horizontal flip, RandomCrop 32 pad 4; checkpoint with the best accuracy on `val_<id>` | OpenOOD v1.5 ([Jingkang50/OpenOOD](https://github.com/Jingkang50/OpenOOD) @ `3c35632`): `configs/pipelines/train/baseline.yml`, `configs/datasets/cifar10/cifar10.yml`, `openood/trainers/base_trainer.py`, `openood/preprocessors/base_preprocessor.py`, `openood/recorders/base_recorder.py` |
| `cifar10_resnet18_supcon` | SupCon, two views, temperature 0.1, MLP projection 512-512-128, batch 1024, lr 0.5, wd 1e-4, 1000 epochs, cosine with 10 warm-up epochs; then a linear probe on the frozen encoder (batch 512, lr 5, 100 epochs, x0.2 at 60/75/90) so that the checkpoint has logits | Khosla et al., NeurIPS 2020; [HobbitLong/SupContrast](https://github.com/HobbitLong/SupContrast) `main_supcon.py`, `main_linear.py`, README |

The ID validation list (`val_cifar10`, 1000 images) is the calibration set of the benchmark. Selecting the checkpoint on it is what OpenOOD does for its official checkpoints; use `select: last` to keep it untouched.

Normalization uses the OpenOOD statistics in every config (also for SupCon, whose repository uses a different CIFAR-10 std), because the extraction backbone `resnet18_32x32` applies them.

## Config format

```yaml
run: my_run                 # results/train/my_run, configs/{extract,benchmarks}/my_run.yaml, my_run.npz
seed: 0
amp: true                   # fp16 autocast + GradScaler on CUDA
select: best                # best (val accuracy in the last stage) | last
data:
  imglists: cifar10
  image_root: images_classic
  train: train_cifar10
  val: val_cifar10          # optional; needed for select: best
  norm: cifar10             # cifar10 | cifar100 | imagenet
  size: 32                  # crop size; evaluation transform = Resize(resize) + CenterCrop(size)
  resize: 32
  cache: true               # keep decoded images in RAM (~150 MB for CIFAR); off for ImageNet-size data
  num_workers: 0
  subsample: null           # optional fixed subset of the training list (same rule as extraction)
model: {name: resnet18_32x32}            # or {name: torchvision, arch: resnet50, weights: null}
stages:                                  # run in order, each with its own loss, optimizer and schedule
  - name: main
    loss: {name: ce}
    epochs: 100
    batch_size: 128
    augment: openood                     # openood | openood_imagenet | supcon | linear | none
    validate: true                       # val accuracy after each epoch
    optimizer: {lr: 0.1, momentum: 0.9, weight_decay: 5.0e-4, nesterov: true}
    schedule: {name: cosine, per: step, min_lr: 1.0e-6}
    # schedule: {name: cosine, per: epoch, min_lr: 5.0e-4, warmup_epochs: 10, warmup_from: 0.01}
    # schedule: {name: step, milestones: [60, 75, 90], gamma: 0.2}
export:
  extract: cifar10_resnet18              # configs/extract template: image sets to extract
  benchmark: cifar10                     # configs/benchmarks template: near/far sets, val_ood
```

After training, `export` writes `configs/extract/<run>.yaml` (the template with `backbone`, `checkpoint`, `options` and `output` replaced) and `configs/benchmarks/<run>.yaml` (the template with `name` and `features` replaced). The benchmark keeps the template's `k` until `--select-k` replaces it with the APS choice of `methods/static/knn.py::select_k` (OpenOOD K grid, capped at the gallery size). The checkpoint path is written relative to the repository root, which `resolve_checkpoint` searches after `CURE_CHECKPOINTS`.

## Models

| `model.name` | Network | Extraction backbone |
|---|---|---|
| `resnet18_32x32` | OpenOOD ResNet-18 for 32x32 inputs (`backbones/openood_nets.py`); requires `size = resize = 32` | `resnet18_32x32` with `norm` |
| `torchvision` | any `torchvision.models.get_model(arch)`; the last `nn.Linear` is replaced by a new `fc`, its input is the feature | `torchvision` with `arch`, `norm`, `size`, `resize` |

`weights` (for example `DEFAULT`) initialises a torchvision model from ImageNet weights. Pretrained weights have seen data close to several OOD sets (Places365, TinyImageNet), so such runs are not comparable with the OpenOOD from-scratch setting. torchvision ResNets downsample by 32, so on 32x32 CIFAR images the last feature map is 1x1; use `size: 224` (upsampling) or a CIFAR network.

To add a network, register a builder that returns a module with `forward(x, return_feature=False) -> logits | (logits, feature)` and an `fc` attribute, plus the name and options of the extraction backbone that rebuilds it from the saved `state_dict` (see [adding_a_backbone.md](adding_a_backbone.md)):

```python
from cure_ood.train.models import ModelSpec, register_model

@register_model("my_net")
def my_net(num_classes, data, width=64):
    return ModelSpec(MyNet(num_classes, width), "my_net", {"width": width, "norm": data["norm"]})
```

The extraction transform of that backbone must equal `train.data.eval_transform(data)`, otherwise the extracted features do not match the training preprocessing (the ID accuracy printed by `cure-ood extract` shows it immediately).

## Losses

| `loss.name` | Objective | Views | Notes |
|---|---|---|---|
| `ce` | cross-entropy on the logits | 1 | |
| `supcon` | supervised contrastive loss (`temperature`, `base_temperature` 0.07, `proj_dim` 128) | 2 | owns the projection head; the loss is computed in fp32 |
| `linear_ce` | cross-entropy on `model.fc` with the encoder frozen (eval mode, no gradient) | 1 | second stage after any representation loss |

A loss is an `nn.Module`; its own parameters (projection heads, prototypes, temperature) are optimised together with the network. To add one, for example CIDER (Ming et al., ICLR 2023; OpenOOD `openood/trainers/cider_trainer.py`):

```python
from cure_ood.train.losses import Objective, register_loss

@register_loss("cider")
class Cider(Objective):
    views = 2

    def __init__(self, feat_dim, num_classes, temperature=0.1, proto_m=0.95, w=2.0, proj_dim=128):
        super().__init__()
        self.head = ...                                   # projection to the hypersphere
        self.register_buffer("prototypes", ...)           # EMA class prototypes, saved in state.pt

    def forward(self, model, x, y):                       # x: [view 1 | view 2], y repeated per view
        _, feats = model(x, return_feature=True)
        ...
        return dis_loss + self.w * comp_loss
```

then use it in a stage followed by a `linear_ce` stage, as in `cifar10_resnet18_supcon.yaml`. A new augmentation is a function registered with `train.data.register_augment(name)` that returns the list of PIL transforms (ToTensor and normalization are appended).

## Cost on an RTX 3050 laptop GPU (6 GB)

Measured with `num_workers: 0`, `cache: true`, AMP, on 5000-image subsets of CIFAR-10 after the first epoch (the first epoch also pays cuDNN autotuning, ~25 s):

| Config | Throughput | Peak GPU memory | Epoch (50k images) | Full run |
|---|---|---|---|---|
| CE, batch 128 | ~1000 images/s | 0.5 GB | ~50 s + 2 s validation | 100 epochs: ~1.5 h |
| SupCon, batch 1024 (2 views) | ~450 images/s | 4.8 GB allocated, 5.2 GB reserved | ~110 s | 1000 epochs: ~31 h |
| linear probe, batch 512 | ~2300 images/s | 0.5 GB | ~22 s + validation | 100 epochs: ~40 min |

Throughput is bound by PIL augmentation in the main process. On Windows, `num_workers: 2` was slower (worker start-up per epoch and pickling of the cached images), so keep 0 unless the images are not cached. SupCon at batch 1024 leaves little room on a 6 GB card: do not run it next to another GPU job, or use `--set stages.0.batch_size=512 --set stages.0.optimizer.lr=0.25` (linear scaling; fewer negatives per batch). 1000 epochs is the paper setting; shorter schedules (e.g. 200-500 epochs) are common in OOD papers that train SupCon/CIDER encoders on CIFAR and are worth trying before committing 31 hours.

## Tests

`tests/test_train.py` trains on a synthetic two-class imglist on CPU (CE with resume, SupCon + linear probe, torchvision round trip, export -> extract -> K selection) in ~20 s. The smoke test on real CIFAR-10 runs only on request:

```bash
CURE_TRAIN_SMOKE=1 pytest tests/test_train.py -k smoke -s   # 1 epoch on 2000 images per config
```

It reads `CURE_SMOKE_IMAGES` (directory containing `images_classic/`) and `CURE_SMOKE_IMGLISTS` (OpenOOD `benchmark_imglist`).
