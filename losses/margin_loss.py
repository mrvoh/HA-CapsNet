from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch


class MarginLoss(nn.Module):
    def __init__(self, m_pos, m_neg, lambda_, binary_class):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_
        self.binary_class = binary_class

    def forward(self, lengths, targets, size_average=False):
        if not self.binary_class:
            t = torch.zeros(lengths.size()).long()
            if targets.is_cuda:
                t = t.cuda()
            t = t.scatter_(1, targets.data.view(-1, 1), 1)
            targets = Variable(t)
        losses = targets.float() * F.relu(self.m_pos - lengths).pow(
            2
        ) + self.lambda_ * (1.0 - targets.float()) * F.relu(lengths - self.m_neg).pow(2)
        return losses.mean() if size_average else losses.sum()
