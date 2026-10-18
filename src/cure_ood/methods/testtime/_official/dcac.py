# Vendored from the official DCAC repository ("DCAC: Dynamic Class-Aware Cache Creates Stronger
# Out-of-Distribution Detectors", AAAI 2026): https://github.com/wyqstan/DCAC, commit 00013c5b234388db7c8f8a85bb50fa57aafa40f2
#   unimodal/ood_method/utils.py: get_entropy, softmax_entropy, update_cache, compute_cache_logits (verbatim)
#   unimodal/ood_method/msp.py:   run_test (loop body verbatim) ; unimodal/configs/msp.yaml (defaults)
# Changes, all marked "ADAPTED":
#   1. run_test reads (logits, features) of one batch instead of calling model.get_feature on images, and returns the
#      per-sample scores instead of AUROC/FPR (metrics are computed by our runner).
#   2. The 4th cache field held the ID/OOD tag of the sample (never used for scoring); it holds the stream index here
#      so that the cache content can be reported (DCR).
#   3. msp and energy scores are returned together (the official code runs the loop once per --method; the cache is
#      the same in both runs because it depends only on logits and features).
import math
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

CONFIG = {"enabled": True, "shot_capacity": 20, "alpha": 0.9, "lower": 95, "topk": 20}  # unimodal/configs/msp.yaml


def get_entropy(loss, num_class):
    max_entropy = math.log2(num_class)
    return loss / max_entropy


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache using FIFO strategy with deque: remove oldest if over capacity."""
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]] + [features_loss[3]]

        if pred not in cache:
            cache[pred] = deque(maxlen=shot_capacity)  # create new deque with fixed capacity

        cache[pred].append(item)  # automatically removes oldest if full


def compute_cache_logits(image_features, cache, alpha, topk=None):
    cache_keys = [item[0] for class_index in sorted(cache.keys()) for item in cache[class_index]]
    cache_values = [item[2] for class_index in sorted(cache.keys()) for item in cache[class_index]]
    cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
    cache_values = torch.cat(cache_values, dim=0)
    topk_values, topk_indices = torch.topk(cache_values, k=topk, dim=1)
    masked_probs = torch.zeros_like(cache_values)
    cache_values = masked_probs.scatter_(1, topk_indices, topk_values)
    affinity = image_features @ cache_keys
    cache_logits = affinity @ cache_values
    return alpha * cache_logits


def train_entropy(train_logits, num_classes):
    """get_train_entropy without the model forward: same formula on the logits of the ID train set."""
    return get_entropy(softmax_entropy(torch.as_tensor(train_logits, dtype=torch.float32)), num_classes).numpy()


@torch.no_grad()
def run_test(logits_all, features_all, id_entropy, cfg, num_classes, batch_size, device):
    """ADAPTED (1-3): msp.run_test over an ordered stream; returns (msp score, energy score, cache trace)."""
    ood_cache = {}
    ood_enabled = cfg['enabled']
    if cfg:
        ood_params = {k: cfg[k] for k in ['shot_capacity', 'alpha', 'lower', 'topk']}
    lower_threshold = np.percentile(id_entropy, ood_params['lower'])
    msp_all, energy_all, trace = [], [], []
    for start in range(0, len(logits_all), batch_size):
        logits = torch.as_tensor(logits_all[start:start + batch_size], dtype=torch.float32, device=device)
        features = torch.as_tensor(features_all[start:start + batch_size], dtype=torch.float32, device=device)
        tags = range(start, start + len(logits))  # ADAPTED (2)
        features = features / features.norm(dim=1, keepdim=True)
        loss = softmax_entropy(logits)
        prob_map = logits.softmax(1)
        pred = torch.argmax(logits, dim=1)
        prop_entropy = get_entropy(loss, num_classes)
        final_logits = logits.clone()
        for i in range(prob_map.shape[0]):
            if ood_enabled and lower_threshold < prop_entropy[i].item():
                update_cache(ood_cache, int(pred[i].item()), [features[i].unsqueeze(0), loss[i].item(), prob_map[i].unsqueeze(0), tags[i]], ood_params['shot_capacity'], True)
        if ood_enabled and ood_cache:
            fuzzy_logits = compute_cache_logits(features, ood_cache, ood_params['alpha'], ood_params['topk'])
            final_logits -= fuzzy_logits
            trace.append((start, np.array([item[3] for c in ood_cache.values() for item in c], dtype=int)))
        msp_all.append(np.max(F.softmax(final_logits, dim=-1).cpu().numpy(), axis=1))
        T = 1
        energy_all.append(T * torch.logsumexp(final_logits / T, dim=1).data.cpu().numpy())
    return np.concatenate(msp_all), np.concatenate(energy_all), trace
