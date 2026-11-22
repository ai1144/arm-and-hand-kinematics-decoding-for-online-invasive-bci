"""Streaming decoder utilities for arm-and-hand kinematics."""

from .data import FeatureConfig, load_dataset, split_trials
from .models import OnlineController, Policy
from .template_policy import OnlineTemplateController, TemplatePolicy

__all__ = [
    "FeatureConfig",
    "OnlineController",
    "OnlineTemplateController",
    "Policy",
    "TemplatePolicy",
    "load_dataset",
    "split_trials",
]
