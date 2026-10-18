# Vendored from the official OODD repository (Yang et al., CVPR 2025):
#   https://github.com/zxk1212/OODD, commit edbb1a32e5fe81e443942156f0a9cafb0297d95b
#   file openood/postprocessors/knn_postprocessor.py (KNNPostprocessor.setup / conf_postprocess)
#   and openood/preprocessors/multicrop_preprocessor.py (MultiRandomCrop).
# The upstream repository ships without a LICENSE file; it builds on OpenOOD (MIT).
#
# Changes, all marked "ADAPTED":
#   1. conf_postprocess shuffled ID+OOD itself (np.random.seed(100)) and split the scores back by label.
#      Here the stream arrives already ordered by our protocol, so the shuffle and the split are removed.
#   2. The priority queue stores the row index of each sample instead of its feature vector, so that the
#      memory content can be reported (DCR). Features are looked up from the same array: identical math.
#   3. Hyper-parameters are function arguments instead of the YAML config (K1, K2, ALPHA, queue_size).
from queue import PriorityQueue

import numpy as np
import torch
import torchvision.transforms as tvs_trans

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10  # noqa: E731  (verbatim, incl. precedence)


def kth_largest_per_column(matrix, k):
    """
    Compute the k-th largest element for each column of a matrix.
    """
    kth_values, _ = torch.kthvalue(matrix, matrix.size(0) - k + 1, dim=0)
    return kth_values.cpu().numpy()


def batched_matrix_multiply(ftrain, second_matrix, K, batch_size=128):
    """
    Compute the product of the feature tensors in batches.
    dimensions: ftrain: (n, d), second_matrix: (p, d), result: (n, p)
    then using the kth largest element in each column as the score
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ftrain_tensor = torch.tensor(ftrain, device=device)
    second_matrix_tensor = torch.tensor(second_matrix, device=device)
    p, _ = second_matrix_tensor.shape

    num_batches = p // batch_size + (p % batch_size > 0)
    res = []
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, p)
        second_batch_tensor = second_matrix_tensor[start:end, :]
        batch_result = ftrain_tensor @ second_batch_tensor.T
        score = kth_largest_per_column(batch_result, K)
        res.append(score)
    return np.concatenate(res, axis=0)


class ScoreData:
    def __init__(self, score, data):
        self.score = score
        self.data = data

    def __lt__(self, other):
        return float(self.score) > float(other.score)


class MultiRandomCrop(torch.nn.Module):
    def __init__(self, num_crop=5, image_size=256):
        super().__init__()
        self.num_crop = num_crop
        self.image_size = image_size

    def forward(self, image):
        cropper = tvs_trans.RandomResizedCrop(
            self.image_size,
            scale=(0.95, 1.0)
        )
        patches = []
        for _ in range(self.num_crop):
            patches.append(cropper(image))
        return torch.stack(patches, 0)


def informative_order(msp_list, label_list):
    """KNNPostprocessor.setup: per class sort by confidence, then interleave classes (verbatim)."""
    unique_classes = torch.unique(label_list)

    class_sorted_indices = []
    max_class_samples = 0

    for cls in unique_classes:
        cls_indices = torch.where(label_list == cls)[0]
        cls_msp = msp_list[cls_indices]
        sorted_cls_indices = torch.argsort(cls_msp, descending=True)
        cls_indices = cls_indices.to(sorted_cls_indices.device)
        sorted_orig_indices = cls_indices[sorted_cls_indices].cpu().numpy()
        class_sorted_indices.append(sorted_orig_indices)
        max_class_samples = max(max_class_samples, len(sorted_orig_indices))

    final_idx = []
    for i in range(max_class_samples):
        for class_idx, sorted_indices in enumerate(class_sorted_indices):
            if i < len(sorted_indices):
                final_idx.append(sorted_indices[i])

    return np.array(final_idx, dtype=int)


@torch.no_grad()
def conf_postprocess(activation_log, idx, all_data, K1=10, K2=5, ALPHA=0.5, queue_size=512, batch_size=512,
                     aux_feature=None, K_prior=None):
    """KNNPostprocessor.conf_postprocess on an already ordered stream `all_data` (rows normalised by `normalizer`).

    Returns (scores, trace): scores are ID scores (larger = more ID) as in the official code;
    trace lists, per batch, the stream indices currently in the queue (aux entries excluded).
    """
    id_train_size = activation_log.shape[0]

    select_idx_len = int(ALPHA * id_train_size)

    ftrain = activation_log[idx[:select_idx_len]]

    queue = PriorityQueue()

    # ADAPTED (2): queue items are row indices into `pool` = [aux_feature; all_data]; aux rows get negative ids.
    n_aux = 0 if aux_feature is None else len(aux_feature)
    pool = all_data if aux_feature is None else np.concatenate([aux_feature, all_data], axis=0)

    if aux_feature is not None:
        # for T-out
        prior_food = aux_feature
        prior_score = batched_matrix_multiply(ftrain, prior_food, K_prior)
        for j in range(prior_food.shape[0]):
            queue.put(ScoreData(prior_score[j], j))
            if queue.qsize() > queue_size:
                queue.get()
        memory_bank = aux_feature[:5]
    else:
        memory_bank = None

    # ADAPTED (1): no internal shuffle; all_data is the stream in protocol order.
    num_batches = all_data.shape[0] // batch_size + (all_data.shape[0] % batch_size > 0)

    scores_list, trace = [], []

    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, all_data.shape[0])
        batch_data = all_data[start:end]
        batch_score = batched_matrix_multiply(ftrain, batch_data, K1)

        for j in range(batch_score.shape[0]):
            queue.put(ScoreData(batch_score[j], n_aux + start + j))
            if queue.qsize() > queue_size:
                queue.get()

        data_list = []
        for item in list(queue.queue):
            data_list.append(item.data)
        new_food = pool[np.array(data_list)]
        if memory_bank is not None:
            new_food = np.concatenate([new_food, memory_bank], axis=0)
        trace.append((start, np.array([d - n_aux for d in data_list if d >= n_aux], dtype=int)))

        ood_batch_score = batched_matrix_multiply(new_food, batch_data, K2)  # 10
        batch_score = batch_score - ood_batch_score

        scores_list.append(batch_score)
    return np.concatenate(scores_list, axis=0), trace
