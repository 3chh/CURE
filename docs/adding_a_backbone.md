# Adding a backbone and producing a feature file

Every benchmark reads one `.npz` feature file. Feature extraction is the only step that touches images and a network. All methods work on the stored arrays.

## Feature file convention

For each set `<set>` (for example `id_train`, `id_test`, `cifar100`, `svhn`):

| Key | Shape / dtype | Required |
|---|---|---|
| `<set>_feats` | `(n, d)` float32, penultimate features, not normalized | yes |
| `<set>_labels` | `(n,)` int64, class id for ID sets, `-1` for OOD | yes |
| `<set>_logits` | `(n, C)` float32 | only if the backbone has a classifier |
| `fc_weight`, `fc_bias` | `(C, d)`, `(C,)` float32 | only if the backbone has a classifier |

Logit-based methods (MSP, Energy, ViM, ...) need `fc_*`. Without them, only feature-based methods run.

## Built-in backbones

| Name | Checkpoint | Notes |
|---|---|---|
| `resnet18_32x32` | `resnet18_cifar10.ckpt`, `resnet18_cifar100.ckpt`, `resnet18_supcon.pt` | OpenOOD v1.5 CIFAR network. Option `norm: cifar10 \| cifar100` selects the normalization statistics. Old `.short.` keys are renamed to `.shortcut.` |
| `resnet18_224x224` | `resnet18_imagenet200.ckpt` | OpenOOD v1.5 ImageNet-200 network |
| `resnet50_in1k` | `resnet50_imagenet1k.pth` | torchvision ResNet-50. Option `amp` (default `true`) keeps the fp16 autocast used for the original features |
| `dinov2_vitb14` | none (torch.hub) | CLS token, fp16 on GPU, no logits |

The number of classes is read from the checkpoint. Checkpoint paths are resolved against `CURE_CHECKPOINTS`, a list of directories separated by `os.pathsep` (default `checkpoints/`).

## Adding a backbone

Register a builder anywhere that is imported before use. To make it built in, put it in `src/cure_ood/backbones/builtin.py`:

```python
import torch
from torchvision import transforms
from cure_ood.backbones import Backbone, register_backbone
from cure_ood.backbones.base import fc_arrays, load_state

@register_backbone("my_net")
def my_net(checkpoint, device, **options) -> Backbone:
    model = MyNet()
    model.load_state_dict(load_state(checkpoint))
    model = model.eval().to(device)
    transform = transforms.Compose([...])  # must match the training preprocessing exactly

    def encode(x):  # x is already on `device`
        feats = model.features(x)
        return feats, model.fc(feats)  # return (feats, None) if there is no classifier

    return Backbone("my_net", model, transform, encode, device, fc_arrays(model.fc))
```

`build_backbone("my_net", checkpoint, device, **options)` then returns an object with `.transform`, `.forward(images) -> (feats, logits | None)` (float32 on CPU) and `.fc`.

When the backbone replaces an earlier extraction script, add a case to `tests/test_backbones.py` that feeds the same synthetic PIL images and random tensors through the old and new paths and checks that they match.

## Producing a feature file

Write `configs/extract/<name>.yaml`:

```yaml
backbone: resnet18_32x32
checkpoint: resnet18_cifar10.ckpt   # relative to CURE_CHECKPOINTS
options: {norm: cifar10}            # passed to the builder
imglists: cifar10                   # $CURE_IMGLISTS/cifar10/<list>.txt, lines "relpath label"
image_root: images_classic          # $CURE_IMAGES/images_classic
roots: {"texture/": images_classic} # optional: per-prefix image directory (longest prefix wins)
dedup: false                        # drop repeated paths (OpenOOD test_imagenet.txt lists each image twice)
skip_missing: false                 # true: drop missing images with a warning instead of failing
seed: 0                             # seed for subsampling
output: my_features.npz             # written to $CURE_FEATURES
batch_size: 512
num_workers: 0
sets:
  id_train: train_cifar10
  id_test: test_cifar10
  svhn: {list: test_svhn, subsample: 10000}   # fixed subset: rng(seed).permutation(n)[:10000]
```

Then run:

```
py -m cure_ood.data.extract <name> [--image-root DIR] [--out FILE] [--device cuda]
```

Default directories: `CURE_IMGLISTS=data/benchmark_imglist` (OpenOOD v1.5 `benchmark_imglist`), `CURE_IMAGES=data/images`, `CURE_FEATURES=data/features`. The script prints the size of each set. For ID sets it also prints top-1 accuracy, so a preprocessing mismatch shows up right away. For reference, the OpenOOD CIFAR-10 ResNet-18 checkpoint gives about 95% on `id_test`.

The shipped configs reproduce the features used in the paper: `cifar10_resnet18`, `cifar10_supcon_resnet18`, `cifar100_resnet18`, `cifar10_dinov2`, `imagenet200_resnet18`, `imagenet1k_resnet50` and `imagenet1k_resnet50_csid`.

## Registering a benchmark

A benchmark pairs a feature file with an evaluation protocol. Add `configs/benchmarks/<name>.yaml`:

```yaml
name: my_bench
description: one line
backbone: my_net
features: my_features.npz   # file in CURE_FEATURES
k: 5                        # kNN neighbour index used by the static scorer
val_ood: val_tin            # optional OOD proxy for hyperparameter selection
near: [cifar100, tin]       # set names in the .npz
far: [mnist, svhn]
# gallery / calibration / id_test default to id_train / id_val / id_test.
# Without an ID train set, split the ID test set by class instead:
# split: {source: id, seed: 0, gallery_frac: 0.8}
```

`Benchmark.load("my_bench")` then loads it. The order of `near` and `far` is part of the stream seed, so reordering them changes the streams.
