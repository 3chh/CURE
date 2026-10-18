"""Phương pháp tĩnh chạy postprocessor gốc của OpenOOD v1.5 qua FeatureNet; cần GPU vì code gốc gọi .cuda()."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

from ..base import Method, Output, register

HERE = Path(__file__).resolve().parent / "_openood"
CLASS_NAMES = {10: "cifar10", 100: "cifar100", 200: "imagenet200", 1000: "imagenet"}  # khóa của info.num_classes_dict

# Module giả cho `import openood.utils.comm` trong base_postprocessor.py (chỉ dùng để tắt thanh tiến trình).
for name in ("openood", "openood.utils"):
    sys.modules.setdefault(name, types.ModuleType(name))
_comm = sys.modules.setdefault("openood.utils.comm", types.ModuleType("openood.utils.comm"))
_comm.is_main_process = lambda: True


class AttrDict(dict):
    def __getattr__(self, k):
        v = self.get(k, None)
        return AttrDict(v) if isinstance(v, dict) else v


class FeatureNet(torch.nn.Module):
    """Phần sau avgpool của ResNet OpenOOD: forward / forward_threshold / get_fc như mạng gốc."""

    def __init__(self, w, b):
        super().__init__()
        self.fc = torch.nn.Linear(w.shape[1], w.shape[0])
        self.fc.weight.data, self.fc.bias.data = torch.as_tensor(w), torch.as_tensor(b)

    def forward(self, x, return_feature=False, return_feature_list=False):
        logits = self.fc(x)
        return (logits, x) if return_feature else logits

    def forward_threshold(self, x, threshold):
        return self.fc(x.clip(max=threshold))

    def get_fc(self):
        return self.fc.weight.cpu().detach().numpy(), self.fc.bias.cpu().detach().numpy()


def loader(feats, labels=None, batch=200):
    """Danh sách batch kiểu ImglistDataset của OpenOOD ({'data', 'label'})."""
    labels = np.zeros(len(feats), int) if labels is None else labels
    return [{"data": torch.as_tensor(np.asarray(feats[s:s + batch], np.float32)),
             "label": torch.as_tensor(np.asarray(labels[s:s + batch]))} for s in range(0, len(feats), batch)]


def recursive_generator(list, n):
    if n == 1:
        results = []
        for x in list[0]:
            k = []
            k.append(x)
            results.append(k)
        return results
    else:
        results = []
        temp = recursive_generator(list, n - 1)
        for x in list[n - 1]:
            for y in temp:
                k = y.copy()
                k.append(x)
                results.append(k)
        return results


class OpenOODMethod(Method):
    """Một postprocessor OpenOOD; điểm OOD = -conf."""
    requires_logits = True
    module = cls_name = cfg_name = ""

    def __init__(self, aps=True, **args):
        cfg = yaml.safe_load((HERE / "configs" / f"{self.cfg_name}.yml").read_text(encoding="utf-8"))["postprocessor"]
        cfg["postprocessor_args"] = {**(cfg.get("postprocessor_args") or {}), **args}
        self.aps, self.cfg = aps, cfg

    def fit(self, bench):
        super().fit(bench)
        import importlib
        mod = importlib.import_module(f".{self.module}", "cure_ood.methods.static._openood.postprocessors")
        n_cls = bench.fc[0].shape[0]
        config = AttrDict(postprocessor=self.cfg, dataset={"name": CLASS_NAMES[n_cls]})
        self.pp = getattr(mod, self.cls_name)(config)
        self.net = FeatureNet(*bench.fc).cuda().eval()
        id_dict = {"train": loader(bench.train_feats, bench.train_labels), "val": loader(bench.cal_feats)}
        val_ood = bench.val_ood_data()[0] if bench.val_ood else None
        ood_dict = {"val": loader(val_ood)} if val_ood is not None else {}
        self.pp.setup(self.net, id_dict, ood_dict)
        self.params = None
        if self.aps and self.cfg.get("APS_mode") and val_ood is not None:
            self.params = self.hyperparam_search(id_dict["val"], ood_dict["val"])
        return self

    def hyperparam_search(self, id_val, ood_val):
        """Evaluator.hyperparam_search: duyệt tổ hợp theo recursive_generator, giữ tổ hợp đầu tiên có AUROC lớn nhất."""
        names = list(self.pp.args_dict.keys())
        combos = recursive_generator([self.pp.args_dict[k] for k in names], len(names))
        max_auroc, final_index = 0, None
        for i, hp in enumerate(combos):
            self.pp.set_hyperparam(hp)
            c_id, c_ood = self.conf(id_val), self.conf(ood_val)
            # compute_all_metrics của OpenOOD: OOD là lớp dương với điểm -conf. Tổ hợp ra NaN (vd ViM dim = 1000 với
            # đặc trưng 512 chiều) thì `auroc > max_auroc` của bản gốc luôn sai, tức bị bỏ qua: làm y như vậy.
            conf = -np.r_[c_id, c_ood]
            auroc = roc_auc_score(np.r_[np.zeros(len(c_id)), np.ones(len(c_ood))], conf) if np.isfinite(conf).all() else np.nan
            if auroc > max_auroc:
                final_index, max_auroc = i, auroc
        self.pp.set_hyperparam(combos[final_index])
        return {k.removesuffix('_list'): v for k, v in zip(names, combos[final_index])}

    @torch.no_grad()
    def conf(self, batches):
        return np.concatenate([self.pp.postprocess(self.net, b["data"].cuda())[1].cpu().numpy() for b in batches])

    def score(self, stream):
        return Output(-self.conf(loader(stream.feats)))


# Tên trong repo → (module, class, file cấu hình) của OpenOOD.
POSTPROCESSORS = {
    "msp": ("base_postprocessor", "BasePostprocessor", "msp"),
    "maxlogit": ("maxlogit_postprocessor", "MaxLogitPostprocessor", "mls"),
    "energy": ("ebo_postprocessor", "EBOPostprocessor", "ebo"),
    "gen": ("gen_postprocessor", "GENPostprocessor", "gen"),
    "mahalanobis": ("mds_postprocessor", "MDSPostprocessor", "mds"),
    "vim": ("vim_postprocessor", "VIMPostprocessor", "vim"),
    "react": ("react_postprocessor", "ReactPostprocessor", "react"),
    "nci": ("nci_postprocessor", "NCIPostprocessor", "nci"),
    "fdbd": ("fdbd_postprocessor", "fDBDPostprocessor", "fdbd"),
}
for _name, (_mod, _cls, _cfg) in POSTPROCESSORS.items():
    register(_name)(type(_cls, (OpenOODMethod,), {"module": _mod, "cls_name": _cls, "cfg_name": _cfg}))


class _CADRefNet(torch.nn.Module):
    """Mạng giả cho CADRef gốc: get_feature trả (logit, đặc trưng) từ đặc trưng có sẵn, giữ thuộc tính fc."""

    def __init__(self, w, b):
        super().__init__()
        self.fc = FeatureNet(w, b).fc

    def get_feature(self, x):
        return self.fc(x), x


@register("cadref")
class CADRef(Method):
    """CADRef gốc (bản đi kèm repo DCAC, commit 00013c5), logit_method Energy như mặc định của code."""
    requires_logits = True

    def fit(self, bench):
        super().fit(bench)
        from ..testtime._official import cadref as official
        w, b = bench.fc
        self.det = official.CADRef(_CADRefNet(w, b).eval(), "cpu")
        feats, logits = official.group_by_prediction(bench.train_logits, bench.train_feats, w.shape[0])
        self.det.set_state(*self.det.get_state(feats, logits))
        return self

    def score(self, stream):
        batches = [(torch.as_tensor(stream.feats[s:s + 200]), None) for s in range(0, len(stream), 200)]
        return Output(-self.det.eval(batches))
