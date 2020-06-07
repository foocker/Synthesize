import torch
import torch.nn as nn
import math
from Synthesize.recognition.representation.face_features import l2_norm


class Arcface(nn.Module):
    # implementation of additive margin softmax losses in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, classnum=51332, s=64., m=0.5):
        super(Arcface, self).__init__()
        self.classnum = classnum
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        # nn.init.xavier_normal_(self.kernel)

        self.m = m  # the margin value, default is 0.5
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1, 0.2397
        # self.mm = math.sin(math.pi - m) * m
        self.threshold = math.cos(math.pi - m)    # -0.8776

    def forward(self, embbedings, label):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel, axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m

        # cond_v = cos_theta - self.threshold
        # cond_mask = cond_v <= 0
        # keep_val = (cos_theta - self.mm)  # when theta not in [0,pi], use cosface instead
        # cos_theta_m[cond_mask] = keep_val[cond_mask]

        cos_theta_m = torch.where((cos_theta - self.threshold) > 0, cos_theta_m, cos_theta-self.mm)

        output = cos_theta * 1.0  # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]

        # one_hot = torch.zeros_like(cos_theta)
        # one_hot.scatter_(1, label.view(-1, 1), 1)
        # output = (one_hot * cos_theta_m) + (1.0 - one_hot) * cos_theta

        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return output