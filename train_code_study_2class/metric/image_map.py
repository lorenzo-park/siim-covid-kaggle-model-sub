# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import Counter
from typing import Any, Callable, Optional, List
from torch import Tensor, tensor
from sklearn.metrics import auc

from torchmetrics import Metric
from torchmetrics.utilities.checks import _check_retrieval_inputs
from torchmetrics.utilities.data import get_group_indexes

import torch
import numpy as np

from metric.utils import calculate_precision_recall

try:
    from torchvision.ops import box_iou
except ModuleNotFoundError:  # pragma: no-cover
    box_iou = None


class ImageLevelMAP(Metric):
    def __init__(
        self,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        samples: bool = True
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn
        )

        self.samples = samples

        if self.samples:
            self.add_state("average_precisions", default=tensor(0.0), dist_reduce_fx="sum")
            self.add_state("total", default=tensor(0.0), dist_reduce_fx="sum")
        else:
            # self.add_state("pred_image_indices", default=[], dist_reduce_fx=None)
            self.add_state("pred_probs", default=[], dist_reduce_fx=None)
            self.add_state("pred_labels", default=[], dist_reduce_fx=None)
            self.add_state("pred_bboxes", default=[], dist_reduce_fx=None)

            # self.add_state("target_image_indices", default=[], dist_reduce_fx=None)
            self.add_state("target_labels", default=[], dist_reduce_fx=None)
            self.add_state("target_bboxes", default=[], dist_reduce_fx=None)

    def update(
        self,
        pred_probs: List[Tensor],
        pred_labels: List[Tensor],
        pred_bboxes: List[Tensor],
        target_labels: List[Tensor],
        target_bboxes: List[Tensor],
    ) -> None:
        """

        """

        if self.samples:
            for score, pred_label, pred_bbox, target_label, target_bbox in zip(pred_probs, pred_labels, pred_bboxes, target_labels, target_bboxes):
                score = score.cpu().detach().numpy()
                pred_label = pred_label.cpu().detach().numpy()
                pred_bbox = pred_bbox.cpu().detach().numpy()
                target_label = target_label.cpu().detach().numpy()
                target_bbox = target_bbox.cpu().detach().numpy()

                score = score[pred_label == 1]
                pred_bbox = pred_bbox[pred_label == 1]
                target_bbox = target_bbox[target_label == 1]

                preds_sorted_idx = np.argsort(score)[::-1]
                pred_bbox = pred_bbox[preds_sorted_idx]

                if target_bbox.shape[0] == 0:
                    continue

                x, y =  calculate_precision_recall(target_bbox, pred_bbox)
                if len(x) >= 2:
                    auc_value = auc(x, y)
                    self.average_precisions += auc_value
                self.total += 1.0
        else:
            # self.pred_image_indices.append(torch.cat(pred_image_indices).long())
            self.pred_probs.append(torch.cat(pred_probs).long())
            self.pred_labels.append(torch.cat(pred_labels).long())
            self.pred_bboxes.append(torch.cat(pred_bboxes).long())

            # self.target_image_indices.append(torch.cat(target_image_indices).long())
            self.target_labels.append(torch.cat(target_labels).long())
            self.target_bboxes.append(torch.cat(target_bboxes).long())

    def compute(self) -> Tensor:
        """
        First concat state `indexes`, `preds` and `target` since they were stored as lists. After that,
        compute list of groups that will help in keeping together predictions about the same query.
        Finally, for each group compute the `_metric` if the number of positive targets is at least
        1, otherwise behave as specified by `self.empty_target_action`.
        """

        if self.samples:
            return self.average_precisions.float() / self.total
        else:
            # pred_image_indices = torch.cat(self.pred_image_indices, dim=0)
            pred_probs = torch.cat(self.pred_probs, dim=0)
            pred_labels = torch.cat(self.pred_labels, dim=0)
            pred_bboxes = torch.cat(self.pred_bboxes, dim=0)

            # target_image_indices = torch.cat(self.target_image_indices, dim=0)
            target_labels = torch.cat(self.target_labels, dim=0)
            target_bboxes = torch.cat(self.target_bboxes, dim=0)

            # pred_index = torch.nonzero((pred_labels == 1))
            # pred_probs = pred_probs[pred_index]
            # pred_bboxes = pred_bboxes[pred_index]
            # target_index = torch.nonzero((target_labels == 1))
            # target_bboxes = target_bboxes[target_index]


            # _, index_sorted = torch.sort(pred_probs)
            # pred_bboxes = pred_bboxes[index_sorted].cpu().detach().numpy()
            # target_bboxes = target_bboxes.cpu().detach().numpy()
            pred_probs = pred_probs.cpu().detach().numpy()
            pred_labels = pred_labels.cpu().detach().numpy()
            pred_bboxes = pred_bboxes.cpu().detach().numpy()
            target_labels = target_labels.cpu().detach().numpy()
            target_bboxes = target_bboxes.cpu().detach().numpy()

            pred_probs = pred_probs[pred_labels == 1]
            pred_bboxes = pred_bboxes[pred_labels == 1]
            target_bboxes = target_bboxes[target_labels == 1]

            preds_sorted_idx = np.argsort(pred_probs)[::-1]
            pred_bboxes = pred_bboxes[preds_sorted_idx]

            x, y =  calculate_precision_recall(target_bboxes, pred_bboxes)

            if len(x) >= 2:
                return auc(x, y)
            else:
                return 0

            # return mean_average_precision(
            #     pred_image_indices,
            #     pred_probs,
            #     pred_labels,
            #     pred_bboxes,
            #     target_image_indices,
            #     target_labels,
            #     target_bboxes,
            #     self.iou_threshold,
            #     self.ap_calculation,
            # )
