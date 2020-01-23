from torch import nn
import torch

class CategoricalCrossEntropyWithSoftmax(nn.Module):
    def __init__(self):
        super(CategoricalCrossEntropyWithSoftmax, self).__init__()
        self.log_softmax = nn.Softmax(dim=1)


    def forward(self, input, target):
        # convert to OHE
        x = input.shape[1]
        target = torch.eye(input.shape[1], requires_grad=True)[target].to(target.device)

        input = torch.clamp(input, 1e-9, 1 - 1e-9)

        # apply log softmax
        input = self.log_softmax(input)

        return -(target * torch.log(input)).sum(dim=1).mean()