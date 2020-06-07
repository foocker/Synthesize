import torch
from itertools import product
from math import ceil
import numpy as np


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        # print(self.feature_maps)
        self.name = 'simple'

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [(j + 0.5) * self.steps[k] / self.image_size[1]]
                    dense_cy = [(i + 0.5) * self.steps[k] / self.image_size[0]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp(max=1,  min=0)
        return output


class PriorBoxBalance(object):
    # for FaceBox network struct
    # 21824 = 32*32*3 + 16*16 + 8*8
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBoxBalance, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = 'balance'

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]    # [[32, 64, 128], [256], [512]]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    if min_size == 32:
                        dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.25, j+0.5, j+0.75]]
                        dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.25, i+0.5, i+0.75]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    elif min_size == 64:
                        dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.5]]
                        dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    else:
                        cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                        cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                        anchors += [cx, cy, s_kx, s_ky]
        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class GeneralAnchors(object):
    def __init__(self, cfg, image_size=None, phase='train', box_normalize=False):
        # x1, y1, x2, y2
        super(GeneralAnchors, self).__init__()
        self.name = 'general'
        self.box_normalize = box_normalize
        self.images_size = image_size
        self.ratios = torch.Tensor(cfg['ratios'])    # np.array([0.5, 1, 2])
        self.scales_per_octave = cfg['scales_per_octave']    # 3
        self.octave_base_scale = 2**2  # 2**2, expand base grid w, h to get scaled anchor
        octave_scales = np.array(
            [2 ** (i / self.scales_per_octave) for i in range(self.scales_per_octave)])
        anchor_scales = octave_scales * self.octave_base_scale
        # self.scales = torch.Tensor(cfg['scales'])    # (2**(0./3), 2**(1.0/3), 2**(2.0/3))
        self.scales = torch.Tensor(anchor_scales)
        self.pyramid_levels = cfg['pyramid_levels']    # [3, 4, 5, 6, 7]  list(return_layer.values()) + 2
        self.strides = [2**x for x in self.pyramid_levels]    # cfg['step'] for get features
        self.base_sizes = [2**x for x in self.pyramid_levels]  # grid w, h can scale use scale and base_size_scale

        self.feature_maps = [[ceil(self.images_size[0]/step), ceil(self.images_size[1]/step)] for step in self.strides]
        self.clip = True

    @property
    def num_base_anchors(self):
        return self.base_anchors.size(0)

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def gen_base_anchors(self, base_size):
        # x1, y1, x2, y2 not x, y, w, h diff from above(ssd), so the encode and decode will...and loss
        # here hold two style
        w, h = base_size, base_size
        x_ctr = 0.5 * (w - 1)
        y_ctr = 0.5 * (h - 1)
        h_ratios = torch.sqrt(self.ratios)
        w_ratios = 1 / h_ratios

        ws = (w * w_ratios[:, None] * self.scales[None, :]).view(-1)    # expand, view(-1)
        hs = (h * h_ratios[:, None] * self.scales[None, :]).view(-1)

        base_anchors = torch.stack(
            [
                x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
            ],
            dim=-1).round()

        return base_anchors

    def grid_anchors(self, featmap_size, base_size, stride=16, device='cuda'):
        base_anchors = self.gen_base_anchors(base_size).to(device)

        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        onefeature_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        onefeature_anchors = onefeature_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return onefeature_anchors

    def forward(self):
        all_anchors = []
        for feature_size, stride in zip(self.feature_maps, self.strides):
            feature_anchors = self.grid_anchors(feature_size, stride, stride)
            if self.clip:
                feature_anchors.clamp(min=0, max=1)

            if self.box_normalize:
                feature_anchors /= self.images_size[0]

            all_anchors.append(feature_anchors)    # when data box not normalize
        return torch.cat(all_anchors, dim=0)    # .unsqueeze(0) for other network, like efficientdet



def test_anchor():

    # test GeneralAnchors
    # from ..configs.detect_config import cfg_detect
    from ..configs.retinageneral_config import cfg_mobile as cfg
    GA = GeneralAnchors(cfg, (640, 640))
    base_anchors = GA.gen_base_anchors(16)
    print(base_anchors)
    all_anchors = GA.forward()
    print(all_anchors.shape, all_anchors.device)    # 80*80*9 + ... + 5*5*9 =
    print(all_anchors[1000:1010, :])

    fab = PriorBox(cfg, (640, 640))
    fab_anchors = fab.forward()
    print(fab_anchors.shape, fab_anchors[1000:1010, :])
