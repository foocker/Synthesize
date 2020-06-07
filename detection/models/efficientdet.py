import torch
from torch import nn
import math
from ..netblocks.netblock import BIFPN
from .retinahead import RetinaHead
from classifier.efficient.efficientnet import EfficientNet
from torchvision.ops import nms
from ..anchors.prior_box import GeneralAnchors
from ..utils.detbox import decode_box, clipbox
# from ..configs.detect_config import cfg
from ..configs.config_eff import cfg_retinahead, cfg_anchor, EFFICIENTDET


MODEL_MAP = {
    'efficientdet-d0': 'efficientnet-b0',
    'efficientdet-d1': 'efficientnet-b1',
    'efficientdet-d2': 'efficientnet-b2',
    'efficientdet-d3': 'efficientnet-b3',
    'efficientdet-d4': 'efficientnet-b4',
    'efficientdet-d5': 'efficientnet-b5',
    'efficientdet-d6': 'efficientnet-b6',
    'efficientdet-d7': 'efficientnet-b6',
}


class EfficientNetD(EfficientNet):
    def __init__(self, blocks_args=None, global_params=None):
        super(EfficientNetD, self).__init__(blocks_args=blocks_args, global_params=global_params)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        p = []
        index = 0
        num_repeat = 0
        # Blocks
        # print('lllllen', len(self._blocks), len(self._blocks_args), self._blocks_args[6].num_repeat)
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            num_repeat = num_repeat + 1
            if num_repeat == self._blocks_args[index].num_repeat:
                num_repeat = 0
                index = index + 1
                p.append(x)
        return p

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        # Convolution layers
        p = self.extract_features(inputs)    # p1 -- p7
        return p

    def get_list_features(self):
        list_feature = []
        for idx in range(len(self._blocks_args)):
            list_feature.append(self._blocks_args[idx].output_filters)

        return list_feature


class BoxClaCom(nn.Module):
    def __init__(self, num_features_in, num_anchors, num_pred, feature_size=256):
        super(BoxClaCom, self).__init__()
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors * num_pred, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.act4(out)
        out = self.output(out)
        return out


class BoxRegHead(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(BoxRegHead, self).__init__()
        self.boxheadcom = BoxClaCom(num_features_in, num_anchors, 4, feature_size=feature_size)

    def forward(self, x):
        out = self.boxheadcom(x)
        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 4)


class ClassPredHead(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, feature_size=256):
        super(ClassPredHead, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.classheadcom = BoxClaCom(num_features_in, num_anchors, num_classes, feature_size)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.classheadcom(x)
        out = self.output_act(out)
        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


# class EfficientDet(nn.Module):
#     def __init__(self, num_classes, levels=3, num_channels=128, network='efficientdet-d0',
#                  is_training=True, threshold=0.5, iou_threshold=0.5):
#         super(EfficientDet, self).__init__()
#         self.efficientnet = EfficientNetD.from_pretrained(MODEL_MAP[network])
#         self.is_training = is_training
#         self.BIFPN = BIFPN(in_channels=self.efficientnet.get_list_features()[2:],
#                            out_channels=256,
#                            num_outs=5)
#         self.boxhead = BoxRegHead()
#         self.classhead = ClassPredHead()
#         self.anchors = GeneralAnchors(cfg, (640, 640))
#
#     def forward(self, x):
#         features = self.efficientnet(x)
#         # p = features
#         # for idx, p in enumerate(p):
#         #     print('P{}: {}'.format(idx + 1, p.size()))
#         features = self.BIFPN(features[2:])
#         regression = torch.cat([self.boxhead(feature) for feature in features], dim=1)
#         classification = torch.cat([self.classhead(feature) for feature in features], dim=1)
#         anchors = self.anchors.forward()
#         if self.is_training:
#             return classification, regression, anchors
#         else:
#             pass


class EfficientDet(nn.Module):
    def __init__(self,
                 num_classes,
                 network='efficientdet-d1',
                 D_bifpn=3,
                 W_bifpn=88,
                 D_class=3,
                 is_training=True,
                 threshold=0.5,
                 iou_threshold=0.5):
        super(EfficientDet, self).__init__()
        self.backbone = EfficientNetD.from_pretrained(MODEL_MAP[network], num_classes=80)
        self.is_training = is_training
        self.neck = BIFPN(in_channels=self.backbone.get_list_features()[-5:],
                                out_channels=W_bifpn,
                                stack=D_bifpn,
                                num_outs=5)
        self.bbox_head = RetinaHead(cfg_retinahead, num_classes=num_classes, in_channels=W_bifpn)
        w, h = EFFICIENTDET[network]['input_size'], EFFICIENTDET[network]['input_size']
        self.anchors = GeneralAnchors(cfg_anchor, image_size=(h, w))    # w, h?
        self.threshold = threshold
        self.iou_threshold = iou_threshold

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.freeze_bn()

    def freeze_bn(self):
        """Freeze BatchNorm layers."""
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def extract_feat(self, img):
        x = self.backbone(img)
        x = self.neck(x[2:])
        return x

    def forward(self, inputs):
        x = self.extract_feat(inputs)
        predictions = self.bbox_head(x)
        cla = torch.cat([o for o in predictions[0]], dim=1)
        reg = torch.cat([o for o in predictions[1]], dim=1)
        anchors = self.anchors.forward()
        if self.is_training:
            return cla, reg, anchors
        else:
            decode_anchors = decode_box(anchors, reg)
            decode_anchors = clipbox(decode_anchors, inputs)
            scores = torch.max(cla, dim=2, keepdim=True)[0]
            scores_over_threshold = (scores > self.threshold)[0, :, 0]

            if scores_over_threshold.sum() == 0:
                return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]
            cla = cla[:, scores_over_threshold, :]
            decode_anchors = decode_anchors[:, scores_over_threshold, :]
            scores = scores[:, scores_over_threshold, :]
            anchors_nms_idx = nms(decode_anchors[0, :, :], scores[0, :, 0], iou_threshold=self.iou_threshold)
            nms_scores, nms_class = cla[0, anchors_nms_idx, :].max(dim=1)
            return [nms_scores, nms_class, decode_anchors[0, anchors_nms_idx, :]]






