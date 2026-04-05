# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from enum import Enum
import logging
from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.regression import R2Score, MeanAbsoluteError
from torchmetrics.utilities.data import dim_zero_cat, select_topk


logger = logging.getLogger("dinov2")


class MetricType(Enum):
    MEAN_ACCURACY = "mean_accuracy"
    MEAN_PER_CLASS_ACCURACY = "mean_per_class_accuracy"
    PER_CLASS_ACCURACY = "per_class_accuracy"
    IMAGENET_REAL_ACCURACY = "imagenet_real_accuracy"
    R2_SCORE = 'r2_score'
    MEAN_ABSOLUTE_ERROR = 'mean_absolute_error'

    @property
    def accuracy_averaging(self):
        return getattr(AccuracyAveraging, self.name, None)

    def __str__(self):
        return self.value


class AccuracyAveraging(Enum):
    MEAN_ACCURACY = "micro"
    MEAN_PER_CLASS_ACCURACY = "macro"
    PER_CLASS_ACCURACY = "none"

    def __str__(self):
        return self.value

def build_reg_metric_mae():
    return MeanAbsoluteError()

def build_reg_metric():
    if metric_type == MetricType.R2_SCORE :
        return R2Score()
    elif metric_type == MetricType.MEAN_ABSOLUTE_ERROR:
        return MeanAbsoluteError()

def build_metric(metric_type: MetricType, *, num_classes: int, ks: Optional[tuple] = None):
    if metric_type.accuracy_averaging is not None:
        return build_topk_accuracy_metric(
            average_type=metric_type.accuracy_averaging,
            num_classes=num_classes,
            ks=(1, 2) if ks is None else ks,
        )
    elif metric_type == MetricType.IMAGENET_REAL_ACCURACY:
        return build_topk_imagenet_real_accuracy_metric(
            num_classes=num_classes,
            ks=(1, 2) if ks is None else ks,
        )

    raise ValueError(f"Unknown metric type {metric_type}")

class cross_entropy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Updates the metric state with predictions and targets.
        Args:
            preds (Tensor): Logits from the model (unnormalized).
            target (Tensor): Ground truth labels (class indices or one-hot encoded).
        """
        # Ensure preds and target are formatted properly
        if preds.ndim != 2 or target.ndim != 1:
            raise ValueError("Expected preds with shape [N, C] and target with shape [N].")

        # Compute Cross Entropy Loss for the batch
        loss = torch.nn.functional.cross_entropy(preds, target, reduction="sum")
        
        # Update states
        self.total_loss += loss
        self.total_samples += target.size(0)

    def compute(self) -> Tensor:
        """
        Computes the average cross-entropy loss over all updates.
        Returns:
            Tensor: Average cross-entropy loss.
        """
        return self.total_loss / self.total_samples


def build_topk_accuracy_metric(average_type: AccuracyAveraging, num_classes: int, ks: tuple = (1, 2)):
    metrics: Dict[str, Metric] = {
        f"top-{k}": MulticlassAccuracy(top_k=k, num_classes=int(num_classes), average=average_type.value) for k in ks}
        
    metrics.update({f"loss": cross_entropy()}) # torchmetric custom 함수로!
    
    return MetricCollection(metrics)


def build_topk_imagenet_real_accuracy_metric(num_classes: int, ks: tuple = (1, 2)):
    metrics: Dict[str, Metric] = {f"top-{k}": ImageNetReaLAccuracy(top_k=k, num_classes=int(num_classes)) for k in ks}
    return MetricCollection(metrics)


class ImageNetReaLAccuracy(Metric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        top_k: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.top_k = top_k
        self.add_state("tp", [], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        # preds [B, D]
        # target [B, A]
        # preds_oh [B, D] with 0 and 1
        # select top K highest probabilities, use one hot representation
        preds_oh = select_topk(preds, self.top_k)
        # target_oh [B, D + 1] with 0 and 1
        target_oh = torch.zeros((preds_oh.shape[0], preds_oh.shape[1] + 1), device=target.device, dtype=torch.int32)
        target = target.long()
        # for undefined targets (-1) use a fake value `num_classes`
        target[target == -1] = self.num_classes
        # fill targets, use one hot representation
        target_oh.scatter_(1, target, 1)
        # target_oh [B, D] (remove the fake target at index `num_classes`)
        target_oh = target_oh[:, :-1]
        # tp [B] with 0 and 1
        tp = (preds_oh * target_oh == 1).sum(dim=1)
        # at least one match between prediction and target
        tp.clip_(max=1)
        # ignore instances where no targets are defined
        mask = target_oh.sum(dim=1) > 0
        tp = tp[mask]
        self.tp.append(tp)  # type: ignore

    def compute(self) -> Tensor:
        tp = dim_zero_cat(self.tp)  # type: ignore
        return tp.float().mean()
