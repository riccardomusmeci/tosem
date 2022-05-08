import torch
from torch import nn
from pytorch_toolbelt.losses import JaccardLoss
# import torch.nn.functional as F

def loss_fn(fn: str, **kwargs) -> nn.Module:
    """returns loss for segmentation model

    Args:
        fn (str): loss function

    Returns:
        nn.Module: loss module
    """
    if fn == "jaccard":
        return JaccardLoss(
            **kwargs
        )
    else:
        print("Only Jaccard is implemented. Quitting.")
        quit()
        
        
# class JaccardLoss(nn.Module):
#     """
#     Implementation of Jaccard loss for image segmentation task.
#     It supports binary, multi-class and multi-label cases.
#     """

#     def __init__(self, mode: str = "multilabel", classes = [0, 1, 2], log_loss=False, from_logits=True, smooth=0, eps=1e-7):
#         """

#         :param mode: Metric mode {'binary', 'multiclass', 'multilabel'}
#         :param classes: Optional list of classes that contribute in loss computation;
#         By default, all channels are included.
#         :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
#         :param from_logits: If True assumes input is raw logits
#         :param smooth:
#         :param eps: Small epsilon for numerical stability
#         """
#         super().__init__()
#         self.mode = mode
#         self.classes = classes
#         self.from_logits = from_logits
#         self.smooth = smooth
#         self.eps = eps
#         self.log_loss = log_loss

#     def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
#         """

#         :param y_pred: NxCxHxW
#         :param y_true: NxHxW
#         :return: scalar
#         """
#         assert y_true.size(0) == y_pred.size(0)

#         if self.from_logits:
#             # Apply activations to get [0..1] class probabilities
#             # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
#             # extreme values 0 and 1
#             if self.mode == "multiclass":
#                 y_pred = y_pred.log_softmax(dim=1).exp()
#             else:
#                 quit()
                

#         bs = y_true.size(0)
#         num_classes = y_pred.size(1)
#         dims = (0, 2)

#         if self.mode == "multiclass":
#             y_true = y_true.view(bs, -1)
#             y_pred = y_pred.view(bs, num_classes, -1)

#             y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
#             y_true = y_true.permute(0, 2, 1)  # H, C, H*W

#         if self.mode == "multilabel":
#             y_true = y_true.view(bs, num_classes, -1)
#             y_pred = y_pred.view(bs, num_classes, -1)

#         scores = soft_jaccard_score(y_pred, y_true.type(y_pred.dtype), smooth=self.smooth, eps=self.eps, dims=dims)

#         if self.log_loss:
#             loss = -torch.log(scores.clamp_min(self.eps))
#         else:
#             loss = 1.0 - scores

#         # IoU loss is defined for non-empty classes
#         # So we zero contribution of channel that does not have true pixels
#         # NOTE: A better workaround would be to use loss term `mean(y_pred)`
#         # for this case, however it will be a modified jaccard loss

#         mask = y_true.sum(dims) > 0
#         loss *= mask.float()

#         if self.classes is not None:
#             loss = loss[self.classes]

#         return loss.mean()
    
    
# def soft_jaccard_score(
#     output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims=None
# ) -> torch.Tensor:
#     """

#     :param output:
#     :param target:
#     :param smooth:
#     :param eps:
#     :param dims:
#     :return:

#     Shape:
#         - Input: :math:`(N, NC, *)` where :math:`*` means
#             any number of additional dimensions
#         - Target: :math:`(N, NC, *)`, same shape as the input
#         - Output: scalar.

#     """
#     print(output.size())
#     print(target.size())
#     quit()
#     assert output.size() == target.size()

#     if dims is not None:
#         intersection = torch.sum(output * target, dim=dims)
#         cardinality = torch.sum(output + target, dim=dims)
#     else:
#         intersection = torch.sum(output * target)
#         cardinality = torch.sum(output + target)

#     union = cardinality - intersection
#     jaccard_score = (intersection + smooth) / (union + smooth).clamp_min(eps)
#     return jaccard_score
