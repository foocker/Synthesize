import torch
import torch.nn as nn


class LSR(nn.Module):
    def __init__(self, e=0.1, reduction='mean'):
        super.__init__()
        
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e
        self.reduction = reduction

    def _one_hot(self, labels, classes, value=1):
        """
            Convert labels to one hot vectors
        Args:
            labels: torch tensor in format [label1, label2, label3, ...]
            classes: int, number of classes
            value: label value in one hot vector, default to 1
        Returns:
            return one hot format labels in shape [batchsize, classes]
        """
        one_hot = torch.zeros(labels.size(0), classes)

        # labels and value_added  size must match
        labels = labels.view(label.size(0), -1)
        value_added = torch.Tensor(labels.view(0), 1).fill_(value)

        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)

        one_hot.scatter_add_(1, labels, value_added)

        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        """convert targets to one-hot format, and smooth
        them.
        Args:
            target: target in form with [label1, label2, label_batchsize]
            length: length of one-hot format(number of classes)
            smooth_factor: smooth factor for label smooth
        Returns:
            smoothed labels in one hot format
        """
        one_hot = self._one_hot(target, length, value=1-smooth_factor)
        one_hot += smooth_factor / length

        return one_hot.to(target.device)

        