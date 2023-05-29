from typing import Optional
from torch import nn, Tensor
import torch.nn.functional as F
from .functional import label_smoothed_nll_loss
import torch

__all__ = ["SoftCrossEntropyLoss"]


class SoftCrossEntropyLoss(nn.Module):
    """
    Drop-in replacement for nn.CrossEntropyLoss with few additions:
    - Support of label smoothing
    """

    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(self, reduction: str = "mean", smooth_factor: float = 0.0, ignore_index: Optional[int] = -100, dim=1):
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # if type(input) == tuple:
        #     # input = torch.cat([fm for fm in input], dim=0)
        #     # print('-------------------------警告tuple类型--------------------------------')
        #     input =input[0]
            # print(type(input[0]))
            # print(type(input[1]))
        # print("显示类型imput",type(input))
        # print("显示类型imput长度", len(input))
        # print("显示类型imput长度", input.shape)
        # print("显示类型target", type(target))
        # print("显示类型target长度",len(target))
        # print("显示类型target长度", target.shape)
        log_prob = F.log_softmax(input, dim=self.dim)
        print("ignore_index",self.ignore_index)

        # 2022.10.15 修改过
        # return label_smoothed_nll_loss(
        #     log_prob,
        #     target,
        #     epsilon=self.smooth_factor,
        #     ignore_index=self.ignore_index,
        #     reduction=self.reduction,
        #     dim=self.dim,
        # )

        return label_smoothed_nll_loss(
            log_prob,
            target,
            epsilon=self.smooth_factor,
            ignore_index=1,
            reduction=self.reduction,
            dim=self.dim,
        )
