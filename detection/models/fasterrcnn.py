import os
import torch
from torchvision.models import resnet
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import misc as misc_nn_ops, MultiScaleRoIAlign
from torchvision.models.detection.backbone_utils import BackboneWithFPN


model_urls = {
    'fasterrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
}


def fasterrcnn_resnetxx_fpnxx(cfg):
    backbone = resnet.__dict__[cfg['backbone_name']](
        pretrained=cfg['backbone_pretrained'],
        norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    # freeze layers
    for name, parameter in backbone.named_parameters():
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)

    # wrapper backbone with fpn
    return_layers = cfg['fpn']['return_layers']
    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2**i for i in range(len(return_layers))]
    out_channels = cfg['fpn']['out_channels']
    backbone_fpn = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)

    anchor_generator = AnchorGenerator(**cfg['anchor_generator'])
    # print(anchor_generator.num_anchors_per_location())

    roi_pooler = MultiScaleRoIAlign(**cfg['box_roi_pool'])
    model = FasterRCNN(backbone_fpn, num_classes=cfg['num_classes'], rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    if os.path.exists(cfg['fasterrcnn_pretrained']):
        state_dict = torch.load(cfg['fasterrcnn_pretrained'])
        model.load_state_dict(state_dict)

    return model


def official_faster(num_classes):
    import torchvision
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator

    # load a pre-trained model for classification and return
    # only the features
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280

    # here mobilenet_v2 just has features, and classifier, has not layer1-layer4, so can'y add fpn

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    return model


def official_instance_segmentation(num_classes):
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


# -----------------------basic practice -------------------------------

# model = fasterrcnn_resnetxx_fpnxx('cfg')
# input = torch.rand(1, 3, 640, 640)
# boxes = torch.rand(6, 4) * 256
# boxes[:, 2:] += boxes[:, :2]
# labels = torch.tensor([1, 1, 2, 3, 1, 1], dtype=torch.int64)
# targets = {'boxes': boxes, 'labels': labels}
# out = model(input, targets)
#
# print(out)


# backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# # print(torchvision.models.mobilenet_v2(pretrained=True).last_channel)
# backbone.out_channels = 1280
# anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
# roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2)
# model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
# model.eval()
# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
# predictions = model(x)
# print(type(predictions), len(predictions), type(predictions[0]), type(predictions[1]))
# print(predictions[0].keys(), predictions[1].keys())


# model, BackboneWithFPN(body:IntermediateLayerGetter, fpn:FeaturePyramidNetwork)

# m = torchvision.models.resnet18(pretrained=True)
# new_m = torchvision.models._utils.IntermediateLayerGetter(m, {'layer1': 'feat1', 'layer3': 'feat2'})
# out = new_m(torch.rand(1, 3, 224, 224))
# print([(k, v.shape) for k, v in out.items()])
#
from collections import OrderedDict
# m = torchvision.ops.FeaturePyramidNetwork([10, 20, 30], 5)
# x = OrderedDict()
# x['feat0'] = torch.rand(1, 10, 64, 64)
# x['feat2'] = torch.rand(1, 20, 16, 16)
# x['feat3'] = torch.rand(1, 30, 8, 8)
# output = m(x)
# print([(k, v.shape) for k, v in output.items()])


# import torch
# import torchvision.models as models
# import torchvision.models.detection.backbone_utils as backbone_utils
#
# backbone = models.resnet50()
# for k, v in backbone.named_children():
#     if k == 'layer2':
#         print(k, v)
#
# return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2}
# in_channels_list = [256, 512, 1024]
# out_channels = 256
# resnet_with_fpn = backbone_utils.BackboneWithFPN(backbone,
#                                                  return_layers, in_channels_list, out_channels)

# for k, v in resnet_with_fpn.named_children():
#     print(k)
#     if k == 'fpn':
#         print(v)
# input = torch.rand(4, 3, 640, 640)
# oupt = resnet_with_fpn(input)
# print(type(oupt))
# for k, v in oupt.items():
#     print(k, v.shape)


# import torchvision.models.detection.rpn as rpn
# import torchvision.models.detection.image_list as image_list
# import torch
#
# # AnchorGenerator
# anchor_generator = rpn.AnchorGenerator()
# concat_box_prediction_layers, AnchorGenerator, RPNHead, RegionProposalNetwork
# filter_proposals,  assign_targets_to_anchors, encode, decode, compute_loss
# print(dir(rpn))
# print(rpn.__dict__.keys())
#
# # ImageList
# batched_images = torch.rand(2, 3, 640, 640)
# image_sizes = [(640, 640)] * 2
# image_list_ = image_list.ImageList(batched_images, image_sizes)
#
# # feature_maps
# feature_maps = [torch.rand([8, 256, 80, 80]), torch.rand(8, 256, 160, 160), torch.rand(8, 256, 320, 320)]
#
# # 80×80×3+160×160×3+320×320×3=403200, aspect_scale=(0.5, 1.0, 2.0)，sizes=(128, 256, 512)
# # anchors
# anchors = anchor_generator(image_list_, feature_maps)
# for anchor in anchors:
#     print(anchor.shape)


# from torchvision.ops.poolers import MultiScaleRoIAlign
# from torchvision.models.detection.roi_heads import RoIHeads
#
#
# m = MultiScaleRoIAlign(['feat1', 'feat3'], 3, 2)
# i = OrderedDict()
# i['feat1'] = torch.rand(1, 5, 64, 64)
# i['feat2'] = torch.rand(1, 5, 32, 32)  # this feature won't be used in the pooling
# i['feat3'] = torch.rand(1, 5, 16, 16)
# boxes = torch.rand(6, 4) * 256; boxes[:, 2:] += boxes[:, :2]
# image_sizes = [(512, 512)]
# output = m(i, [boxes], image_sizes)
# print(boxes, output.shape)
#
# -----------------------basic practice -------------------------------


