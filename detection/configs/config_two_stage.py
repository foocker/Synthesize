from data.coco_vison import get_transform, collate_fn

cfg_two_stage = {
    'root': '*/dataset/VOC/VOCdevkit/VOC2007/JPEGImages',
    'annFile': '*/dataset/VOC/VOCdevkit/VOC2007/AnnotationsJson/voc2007_to_coco.json',
    'transform': get_transform(),
    'batch_size': 8,
    'collate_fn': collate_fn
}

Struct_Component_Cfg = {
    'backbone_pretrained': True,
    'backbone_name': 'resnet50',
    'fpn': {'return_layers': {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}, 'out_channels': 256},
    'num_classes': 21,    # (including the background).
    'anchor_generator': {'sizes': ((32,), (64,), (128, ), (256,), (512, )), 'aspect_ratios': ((0.5, 1.0, 2.0),)*5},
    'box_roi_pool': {'featmap_names': [0, 1, 2, 3], 'output_size': 7, 'sampling_ratio': 2},    # Consistent with fpn
    'fasterrcnn_pretrained': ''    # weight path
}
# Struct_Component_Cfg = {
#     'backbone_pretrained': True,
#     'backbone_name': 'resnet18',
#     'fpn': {'return_layers': {'layer1': 0, 'layer2': 1, 'layer3': 2}, 'out_channels': 256},
#     'num_classes': 21,    # (including the background).
#     'anchor_generator': {'sizes': ((32,), (64,), (128, ), (256,)), 'aspect_ratios': ((0.5, 1.0, 2.0),)*4},
#     'box_roi_pool': {'featmap_names': [0, 1, 2], 'output_size': 7, 'sampling_ratio': 2},    # Consistent with fpn
#     'fasterrcnn_pretrained': ''    # weight path
# }
