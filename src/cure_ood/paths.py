# -*- coding: utf-8 -*-
"""
Centralized, platform-independent path resolution for CURE.
Guarantees 100% relative path resolution across Local (Windows/Linux), Kaggle, Colab, and Clusters.
"""
from __future__ import annotations

import os
from pathlib import Path

# Repository root is 2 levels up from src/cure_ood
REPO_ROOT = Path(__file__).resolve().parents[2]

def get_repo_root() -> Path:
    return REPO_ROOT

def get_data_dir() -> Path:
    p = os.environ.get("CURE_DATA")
    return Path(p).resolve() if p else REPO_ROOT / "data"

def get_features_dir() -> Path:
    p = os.environ.get("CURE_FEATURES")
    return Path(p).resolve() if p else get_data_dir() / "features"

def get_checkpoints_dir() -> Path:
    p = os.environ.get("CURE_CHECKPOINTS")
    return Path(p).resolve() if p else REPO_ROOT / "results" / "train"

def get_results_dir() -> Path:
    p = os.environ.get("CURE_RESULTS")
    return Path(p).resolve() if p else REPO_ROOT / "results"

def get_figures_dir() -> Path:
    p = os.environ.get("CURE_OUTPUT")
    return Path(p).resolve() if p else REPO_ROOT / "deliverables" / "figures"

