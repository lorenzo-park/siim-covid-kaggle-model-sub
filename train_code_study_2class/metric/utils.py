from numba import jit

import numpy as np


@jit(nopython=True)
def calculate_iou(gt, pr, form='pascal_voc') -> float:
    """Calculates the Intersection over Union.

    Args:
        gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box
        pr: (np.ndarray[Union[int, float]]) coordinates of the prdected box
        form: (str) gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        (float) Intersection over union (0.0 <= iou <= 1.0)
    """
    if form == 'coco':
        gt = gt.copy()
        pr = pr.copy()

        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]

    # Calculate overlap area
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1

    if dx < 0:
        return 0.0

    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1

    if dy < 0:
        return 0.0

    overlap_area = dx * dy

    # Calculate union area
    union_area = (
            (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) +
            (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) -
            overlap_area
    )

    return overlap_area / union_area


@jit(nopython=True)
def find_best_match(gts, pred, pred_idx, threshold = 0.5, form = 'pascal_voc', ious=None) -> int:
    """Returns the index of the 'best match' between the
    ground-truth boxes and the prediction. The 'best match'
    is the highest IoU. (0.0 IoUs are ignored).

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        pred: (List[Union[int, float]]) Coordinates of the predicted box
        pred_idx: (int) Index of the current predicted box
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (int) Index of the best match GT box (-1 if no match above threshold)
    """
    best_match_iou = -np.inf
    best_match_idx = -1

    for gt_idx in range(len(gts)):

        if gts[gt_idx][0] < 0:
            # Already matched GT-box
            continue

        iou = -1 if ious is None else ious[gt_idx][pred_idx]

        if iou < 0:
            iou = calculate_iou(gts[gt_idx], pred, form=form)

            if ious is not None:
                ious[gt_idx][pred_idx] = iou

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return best_match_idx


@jit(nopython=True)
def calculate_precision(gts, preds, threshold = 0.5, form = 'pascal_voc', ious=None) -> float:
    """Calculates precision for GT - prediction pairs at one threshold.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (float) Precision
    """
    n = len(preds)
    tp = 0
    fp = 0

    # for pred_idx, pred in enumerate(preds_sorted):
    for pred_idx in range(n):

        best_match_gt_idx = find_best_match(gts, preds[pred_idx], pred_idx,
                                            threshold=threshold, form=form, ious=ious)

        if best_match_gt_idx >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            tp += 1
            # Remove the matched GT box
            gts[best_match_gt_idx] = -1

        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            fp += 1

    # False negative: indicates a gt box had no associated predicted box.
    fn = (gts.sum(axis=1) > 0).sum()

    return tp / (tp + fp), tp / (tp + fn) # Precision, Recall


# @jit(nopython=True)
# def calculate_precision_recall(gts, preds, md=True, thresholds = (0.5,), form = 'pascal_voc') -> float:
#     """Calculates image precision.

#     Args:
#         gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
#         preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
#                sorted by confidence value (descending)
#         thresholds: (float) Different thresholds
#         form: (str) Format of the coordinates

#     Return:
#         (float) Precision
#     """
#     ious = np.ones((len(gts), len(preds))) * -1
#     # ious = None

#     x = []
#     y = []
#     for threshold in thresholds:
#         precision, recall = calculate_precision(gts.copy(), preds, threshold=threshold,
#                                                      form=form, ious=ious)
#         x.append(recall)
#         y.append(precision)
#     if md:
#         y = monotone_decrease(y)
#     return x, y


@jit(nopython=True)
def monotone_decrease(y):
    new_y = []
    for idx in range(len(y)):
        new_y.append(max(y[idx:]))
    return new_y


@jit(nopython=True)
def calculate_precision_recall(gts, preds, thresholds = 0.5, form = 'pascal_voc'):
    """Calculates image precision.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        thresholds: (float) Different thresholds
        form: (str) Format of the coordinates

    Return:
        (float) Precision
    """

    precisions = []
    recalls = []
    for i in range(len(preds)+1):
        tp = 0
        fp = 0
        fn = np.ones(len(gts))

        for pred in preds[:i]:
            matched = False
            for idx, gt in enumerate(gts):
                iou = calculate_iou(gt, pred)

                if iou >= thresholds:
                    if matched is False:
                        tp += 1
                        matched = True
                    else:
                        fp += 1
                    fn[idx] = 0
                else:
                    fp += 1
        fn = np.sum(fn)

        if fp + tp > 0:
            prec = tp / (fp + tp)
        else:
            prec = 1
        if fn + tp > 0:
            rec = tp / (fn + tp)
        else:
            rec = 0
        recalls.append(rec)
        precisions.append(prec)
    return recalls, monotone_decrease(precisions)
