import torch
from torch import Tensor


class KLDivergence():
    def __init__(self, reduction: str = "mean", **kwargs):
        super().__init__(**kwargs)
        assert reduction in ["sum", "mean", "none"]
        self.reduction = reduction

    def __call__(self, means: Tensor, logvars: Tensor) -> Tensor:
        kl_cost = -0.5 * (1.0 + logvars - means**2 - logvars.exp())
        kl_cost = torch.sum(kl_cost, 1)
        if self.reduction == "none":
            return kl_cost
        elif self.reduction == "sum":
            return torch.sum(kl_cost)
        else:
            return torch.mean(kl_cost, 0)
