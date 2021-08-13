import torch
import torch.nn.functional as F

"""
CREDIT: https://github.com/peterliht/knowledge-distillation-pytorch/blob/ef06124d67a98abcb3a5bc9c81f7d0f1f016a7ef/mnist/distill_mnist.py
"""
def distillation_loss(outputs, targets, teacher_scores, temperature, alpha):
    # return F.kl_div((F.log_softmax(outputs/temperature), F.softmax(teacher_scores/temperature)) * (temperature*temperature * 2.0 * alpha) + F.cross_entropy(outputs, targets) * (1. - alpha))
    # distill_loss = F.kl_div(
    #     F.log_softmax(outputs/teacher_scores, dim=1),
    #     F.softmax(targets/teacher_scores, dim=1), reduction='batchmean'
    # )
    distill_loss = F.binary_cross_entropy_with_logits(
        outputs,
        targets
    )
    cls_loss = F.cross_entropy(outputs, torch.argmax(targets, dim=1))
    return distill_loss * alpha + cls_loss * (1 - alpha)
