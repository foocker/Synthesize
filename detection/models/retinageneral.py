import torch.nn as nn
from detection.netblocks.netblock import FPN, FPNGeneral, SSH, MobileNetV1
import torch
from torchvision.models import _utils


class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anhors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anhors*2, (1, 1), 1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


class BoxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(BoxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors*4, (1, 1), 1, 0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors*10, (1, 1), 1, 0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 10)


class RetinaGeneral(nn.Module):
    def __init__(self, cfg=None, phase='train', scene='face'):
        # scene: face, ocr, general
        assert scene in ('face', 'ocr', 'general'), 'scene:{} is not support'.format(scene)
        super(RetinaGeneral, self).__init__()
        self.phase = phase
        self.scene = scene
        self.num_fpn = len(cfg['return_layers'])   # 3, 4: small, mid, big
        # self.anchor_num = len(cfg['ratios']) * len(cfg['scales'])    # diverse: face, general, word, different scene
        self.anchor_num = 2   # diverse: face, general, word, different scene
        self.in_channels = cfg['in_channel']    # according to backbone structure, the first return layer channel
        self.return_layer = cfg['return_layers']    # from backbone scale layers, what you want extract
        # self.in_channels_list = [self.in_channels*2**i for i in list(self.return_layer.values())]   # correspond the fpn
        self.out_channels = cfg['out_channel']    # setting parameters
        # self.fpn = cfg['fpn'](self.in_channels_list, self.out_channels)    # fpn, bifpn and so on

        # assert self.num_fpn == len(self.in_channels_list), 'num fpn should equal to num return layer'

        # self.ssh_list = [SSH(self.out_channels, self.out_channels).to('cuda') for _ in range(self.num_fpn)]    # just trick
        # self.ssh_use = True

        backbone = None
        # should change sparse form cfg, not judgment
        if cfg['name'] == 'mobilenet0.25_retinaface_general':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load('', map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'resnet50_ratinaface':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])

        self.body = _utils.IntermediateLayerGetter(backbone, self.return_layer)    # add
        # self.in_channels = ?    # calculate from input tensor

        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list, out_channels)    # add bifpn
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.BoxHead = self._make_bbox_head(fpn_num=self.num_fpn, inchannels=self.out_channels, anchor_num=self.anchor_num)
        self.ClassHead = self._make_class_head(fpn_num=self.num_fpn, inchannels=self.out_channels, anchor_num=self.anchor_num)
        self.LandmarkHead = None
        if self.scene == 'face':
            self.LandmarkHead = self._make_landmark_head(fpn_num=self.num_fpn, inchannels=self.out_channels, anchor_num=self.anchor_num)

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=9):
        bboxhead = nn.ModuleList()
        for _ in range(fpn_num):
            bboxhead.append(BoxHead(inchannels, anchor_num))
        return bboxhead

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=9):
        classhead = nn.ModuleList()
        for _ in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=9):
        landmarkhead = nn.ModuleList()
        for _ in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def forward(self, inputs):
        out = self.body(inputs)

        fpn = self.fpn(out)

        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh2(fpn[2])
        features = [feature1, feature2, feature3]

        # features = [self.ssh_list[i](fpn[i]) if self.ssh_use else fpn[i] for i in range(self.num_fpn)]

        box_regressions = torch.cat([self.BoxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        if self.LandmarkHead is not None:
            ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)
        else:
            ldm_regressions = None

        if self.phase == 'train':
            output = (box_regressions, classifications, ldm_regressions)
        else:
            output = (box_regressions, nn.functional.softmax(classifications, dim=-1), ldm_regressions)
        # print(output[0].shape, output[1].shape, output[2].shape)
        # torch.Size([32, 16800, 4]) torch.Size([32, 16800, 2]) torch.Size([32, 16800, 10])
        return output
