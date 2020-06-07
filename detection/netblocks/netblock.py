import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# just combine classic basic operator to create functional module
def conv_bn(inc, ouc, stride=1, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inc, ouc, 3, stride, 1, bias=False),
        nn.BatchNorm2d(ouc),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


def conv_bn_no_relu(inc, ouc, stride):
    return nn.Sequential(
        nn.Conv2d(inc, ouc, 3, stride, 1, bias=False),
        nn.BatchNorm2d(ouc)
    )


def conv_bn1x1(inc, ouc, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inc, ouc, 1, stride, 0, bias=False),
        nn.BatchNorm2d(ouc),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


def conv_dw(inc, ouc, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inc, inc, 3, stride, 1, groups=inc, bias=False),
        nn.BatchNorm2d(inc),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),

        nn.Conv2d(inc, ouc, 1, 1, 0, bias=False),
        nn.BatchNorm2d(ouc),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


class BasicConv2d(nn.Module):
    def __init__(self, inc, ouc, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(inc, ouc, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(ouc, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


# classic functional module, but just for some special network, not auto dynamic
class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.branch1x1 = BasicConv2d(128, 32, kernel_size=1, padding=0)
        self.branch1x1_2 = BasicConv2d(128, 32, kernel_size=1, padding=0)
        self.branch3x3_reduce = BasicConv2d(128, 24, kernel_size=1, padding=0)
        self.branch3x3 = BasicConv2d(24, 32, kernel_size=3, padding=1)
        self.branch3x3_reduce_2 = BasicConv2d(128, 24, kernel_size=1, padding=0)
        self.branch3x3_2 = BasicConv2d(24, 32, kernel_size=3, padding=1)
        self.branch3x3_3 = BasicConv2d(32, 32, kernel_size=3, padding=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch1x1_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch1x1_2 = self.branch1x1_2(branch1x1_pool)

        branch3x3_reduce = self.branch3x3_reduce(x)
        branch3x3 = self.branch3x3(branch3x3_reduce)

        branch3x3_reduce_2 = self.branch3x3_reduce_2(x)
        branch3x3_2 = self.branch3x3_2(branch3x3_reduce_2)
        branch3x3_3 = self.branch3x3_3(branch3x3_2)

        outputs = [branch1x1, branch1x1_2, branch3x3, branch3x3_2]    # 32*4=128
        return torch.cat(outputs, 1)


class CRelu(nn.Module):
    def __init__(self, inc, ouc, **kwargs):
        super(CRelu, self).__init__()
        self.conv = nn.Conv2d(inc, ouc, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(ouc, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.cat([x, -x], 1)
        x = F.relu(x, inplace=True)
        return x


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        leaky = 0
        if out_channels <= 64:
            leaky = 0.1
        self.output1 = conv_bn1x1(in_channels_list[0], out_channels, 1, leaky)
        self.output2 = conv_bn1x1(in_channels_list[1], out_channels, 1, leaky)
        self.output3 = conv_bn1x1(in_channels_list[2], out_channels, 1, leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, input):
        input = list(input.values())
        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode='nearest')
        output2 = output2 + up3
        output2 = self.merge1(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode='nearest')
        output1 = output1 + up2
        output1 = self.merge2(output1)

        out = [output1, output2, output3]
        return out


class FPNGeneral(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPNGeneral, self).__init__()
        leaky = 0
        if out_channels <= 64:
            leaky = 0.1
        self.outputs = [conv_bn1x1(in_channels, out_channels, 1, leaky).to('cuda') for in_channels in in_channels_list]
        self.merges = [conv_bn(out_channels, out_channels, leaky=leaky).to('cuda') for _ in range(len(self.outputs) - 1)]

    def forward(self, input):
        input = list(input.values())
        outputs = [self.outputs[i](input[i]) for i in range(len(input))]
        lenght = len(self.merges)    # top-to-down, down-to-up, other
        for i in range(lenght):
            upi = F.interpolate(outputs[i+1], size=[outputs[i].size(2), outputs[i].size(3)], mode='nearest')
            outputs[i] = self.merges[lenght-i-1](upi + outputs[i])
        return outputs


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky = 0.1),    # 3
            conv_dw(8, 16, 1),   # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),   # 43 + 16 = 59
            conv_dw(128, 128, 1),  # 59 + 32 = 91
            conv_dw(128, 128, 1),  # 91 + 32 = 123
            conv_dw(128, 128, 1),  # 123 + 32 = 155
            conv_dw(128, 128, 1),  # 155 + 32 = 187
            conv_dw(128, 128, 1),  # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2),  # 219 +3 2 = 241
            conv_dw(256, 256, 1),  # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x


class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if out_channel <= 64:
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


# for efficient net or others, rewrite some basic operator to create functional module
def conv_ws_2d(input, weight, bias=None, stride=1, padding=0, dilation=1,
               groups=1, eps=1e-5):
    c_in = weight.size(0)
    weight_flat = weight.view(c_in, -1)
    mean = weight_flat.mean(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    std = weight_flat.std(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    weight = (weight - mean) / (std + eps)
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


class ConvWS2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, eps=1e-5):
        super(ConvWS2d, self).__init__(in_channels, out_channels, kernel_size,
                                       stride=stride, padding=padding, dilation=dilation,
                                       groups=groups, bias=bias)
        self.eps = eps

    def forward(self, x):
        return conv_ws_2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation,
                          self.groups, self.eps)

conv_cfg = {
    'Conv': nn.Conv2d,
    'ConvWS': ConvWS2d
}


def build_conv_layer(cfg, *args, **kwargs):
    """ Build convolution layer
    Args:
        cfg (None or dict): cfg should contain:
            type (str): identify conv layer type.
            layer args: args needed to instantiate a conv layer.
    Returns:
        layer (nn.Module): created conv layer
    """
    if cfg is None:
        cfg_ = dict(type='Conv')
    else:
        assert isinstance(cfg, dict) and 'type' in cfg
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in conv_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        conv_layer = conv_cfg[layer_type]

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer


norm_cfg = {
    # format: layer_type: (abbreviation, module)
    'BN': ('bn', nn.BatchNorm2d),
    'SyncBN': ('bn', nn.SyncBatchNorm),
    'GN': ('gn', nn.GroupNorm),
    # and potentially 'SN'
}


def build_norml_layer(cfg, num_features, postfix=''):
    """ Build normalization layer
    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.
    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN':
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer


class ConvModule(nn.Module):
    """
    A conv block that contains conv/norm/activation layers.
    args:
        bias(bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.

        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
        activation (str or None): Activation type, "ReLU" by default.
        inplace (bool): Whether to use inplace mode for activation.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias='auto', conv_cfg=None, norm_cfg=None,
                 activation='relu', inplace=True, order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_activatation = activation is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias

        self.conv = build_conv_layer(conv_cfg, in_channels, out_channels, kernel_size, stride=stride,
                                     padding=padding, dilation=dilation, groups=groups, bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_norm:
            # norm layer after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norml_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)

        if self.with_activatation:
            if self.activation not in ['relu']:
                raise ValueError('{} is currently not supported'.format(self.activation))
            if self.activation == 'relu':
                self.activate = nn.ReLU(inplace=inplace)    # activation just cfg str

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activatation:
                x = self.activate(x)
        return x


class BIFPN(nn.Module):
    def __init__(self, in_channels, out_channels, num_outs, start_level=0, end_level=-1, stack=1,
                 add_extra_convs=False, extra_convs_on_inputs=True, relu_before_extra_convs=False,
                 no_norm_on_lateral=False, conv_cfg=None, norm_cfg=None, activation=None):
        super(BIFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.stack = stack

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.stack_bifpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False)
            self.lateral_convs.append(l_conv)

        for ii in range(stack):
            self.stack_bifpn_convs.append(BiFPNModule(channels=out_channels,
                                                      levels= self.backbone_end_level-self.start_level,
                                                      conv_cfg=conv_cfg,
                                                      norm_cfg=norm_cfg,
                                                      activation=activation))
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # part 1: build top-down and down-top path with stack
        used_backbone_levels = len(laterals)
        for bifpn_module in self.stack_bifpn_convs:
            laterals = bifpn_module(laterals)
        outs = laterals
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[0](orig))
                else:
                    outs.append(self.fpn_convs[0](outs[-1]))
                for i in range(1, self.num_outs - used_backbone_levels):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        return tuple(outs)


class BiFPNModule(nn.Module):
    def __init__(self,
                 channels,
                 levels,
                 init=0.5,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 eps=1e-4):
        super(BiFPNModule, self).__init__()
        self.eps = eps
        self.activation = activation
        self.levels = levels
        self.bifpn_convs = nn.ModuleList()
        # weighted
        self.w1 = nn.Parameter(torch.Tensor(2, levels).fill_(init))
        self.relu1 = nn.ReLU()
        self.w2 = nn.Parameter(torch.Tensor(3, levels - 2).fill_(init))
        self.relu2 = nn.ReLU()
        for jj in range(2):
            for i in range(self.levels-1):  # 1,2,3
                fpn_conv = nn.Sequential(
                    ConvModule(
                        channels,
                        channels,
                        3,
                        padding=1,
                        groups=channels,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        activation=self.activation,
                        inplace=False),
                    ConvModule(
                        channels,
                        channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        activation=self.activation,
                        inplace=False))
                self.bifpn_convs.append(fpn_conv)

    def forward(self, inputs):
        assert len(inputs) == self.levels
        # build top-down and down-top path with stack
        levels = self.levels
        # w relu
        w1 = self.relu1(self.w1)
        w1 /= torch.sum(w1, dim=0) + self.eps  # normalize
        w2 = self.relu2(self.w2)
        w2 /= torch.sum(w2, dim=0) + self.eps
        # build top-down
        idx_bifpn = 0
        # pathtd = inputs copy is wrong
        pathtd = inputs
        inputs_clone = []
        for in_tensor in inputs:
            inputs_clone.append(in_tensor.clone())    # detech
        for i in range(levels - 1, 0, -1):
            pathtd[i - 1] = (w1[0, i-1]*pathtd[i - 1] + w1[1, i-1]*F.interpolate(
                pathtd[i], scale_factor=2, mode='nearest'))/(w1[0, i-1] + w1[1, i-1] + self.eps)
            pathtd[i - 1] = self.bifpn_convs[idx_bifpn](pathtd[i - 1])
            idx_bifpn += 1

        # build down-top
        for i in range(0, levels - 2, 1):
            pathtd[i + 1] = (w2[0, i] * pathtd[i + 1] + w2[1, i] * F.max_pool2d(pathtd[i], kernel_size=2) + w2[2, i] * \
                            inputs_clone[i + 1])/(w2[0, i] + w2[1, i] + w2[2, i] + self.eps)
            pathtd[i + 1] = self.bifpn_convs[idx_bifpn](pathtd[i + 1])
            idx_bifpn += 1

        pathtd[levels - 1] = (w1[0, levels-1] * pathtd[levels - 1] + w1[1, levels-1] * F.max_pool2d(
            pathtd[levels - 2], kernel_size=2))/(w1[0, levels-1] + w1[1, levels-1] + self.eps)
        pathtd[levels - 1] = self.bifpn_convs[idx_bifpn](pathtd[levels - 1])
        return pathtd


# some init function
def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


def bias_init_with_prob(prior_prob):
    """ initialize conv/fc bias value according to giving probablity"""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init