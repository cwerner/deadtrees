# alternative implementation to default GDL

import torch


class GeneralizedDiceLoss(torch.nn.Module):
    def __init__(self):
        super(GeneralizedDiceLoss, self).__init__()

    def forward(self, inp, targ):
        inp = inp.contiguous().permute(0, 2, 3, 1)
        targ = targ.contiguous().permute(0, 2, 3, 1)

        w = torch.zeros((targ.shape[-1],))
        w = 1.0 / (torch.sum(targ, (0, 1, 2)) ** 2 + 1e-9)

        numerator = targ * inp
        numerator = w * torch.sum(numerator, (0, 1, 2))
        numerator = torch.sum(numerator)

        denominator = targ + inp
        denominator = w * torch.sum(denominator, (0, 1, 2))
        denominator = torch.sum(denominator)

        dice = 2.0 * (numerator + 1e-9) / (denominator + 1e-9)

        return 1.0 - dice
