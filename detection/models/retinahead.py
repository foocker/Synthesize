from torch import nn
from functools import partial
from six.moves import map, zip
from ..netblocks.netblock import ConvModule, normal_init, bias_init_with_prob


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


class RetinaHead(nn.Module):
    """
     An anchor-based head used in [1].
     The head contains two subnetworks. The first classifies anchor boxes and
     the second regresses deltas for the anchors.
     References:
         .. [1]  https://arxiv.org/pdf/1708.02002.pdf
     Here has no loss in This Head Class, so it isn't head....
    """
    def __init__(self, cfg, num_classes, in_channels):
        super(RetinaHead, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = cfg['feat_channels']
        self.anchor_ratios = cfg['anchor_ratios']
        self.anchor_scales = cfg['anchor_scales']
        self.conv_cfg = cfg['conv_cfg']    # for basic convblock
        self.norm_cfg = cfg['norm_cfg']
        self.stacked_convs = cfg['stacked_convs']    # repeat times conv+bn+relu for head
        self.cls_out_channels = num_classes
        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)
        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)
        self.output_act = nn.Sigmoid()

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        cls_score = self.retina_cls(cls_feat)
        cls_score = self.output_act(cls_score)
        # out is B x C x W x H, with C = n_classes + n_anchors
        cls_score = cls_score.permute(0, 2, 3, 1)
        batch_size, width, height, channels = cls_score.shape
        cls_score = cls_score.view(batch_size, width, height, self.num_anchors, self.num_classes)
        cls_score = cls_score.contiguous().view(x.size(0), -1, self.num_classes)

        bbox_pred = self.retina_reg(reg_feat)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1)
        bbox_pred = bbox_pred.contiguous().view(bbox_pred.size(0), -1, 4)
        return cls_score, bbox_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)


