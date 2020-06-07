from ..anchors import PriorBoxBalance, PriorBox, GeneralAnchors
from ..models import RetinaFace, FaceBoxes, RetinaGeneral
from ..netblocks.netblock import FPN,  BIFPN, FPNGeneral


cfg_test = {
    'confidence_threshold': 0.1,
    'top_k': 50,
    'nms_threshold': 0.3,
    'keep_top_k': 25,
    'save_images': True,
    'vis_thres': 0.8,  # visualization_threshold
    'origin_size': True,  # Whether use origin image size to evaluate
    'trained_model': '/*/Synthesize/weights_detect/*.pth',  # mobilenet0.25_retinaface_general_Final.pth
    'save_folder': '',      # format report files for evaluate
    'result_folder': '',    # draw box images

    # 'trained_model': '/*/Synthesize/weights_detect/facebox_Final_.pth',
    # 'confidence_threshold': 0.02,
    # 'top_k': 3000,
    # 'nms_threshold': 0.4,
    # 'keep_top_k': 400,
    # 'save_images': True,
    # 'vis_thres': 0.5,  # visualization_threshold
    # 'origin_size': True,  # Whether use origin image size to evaluate
    # 'save_folder': '/*/dataset/faces/face_detect/wider_face/WIDER/WIDER_val/Pred_txt_facebox/',
    # 'result_folder': '/*/dataset/faces/face_detect/wider_face/WIDER/WIDER_val/result_facebox/',

    'test_dataset': '/*/dataset/',
    'test_label': '/*/dataset/xx.json',

}


cfg_detect = {
    'anchor_sizes': '',
    'aspect_ratios': '',
    'anchor_strides': '',
    'straddle_thresh': '',
    'octave': '',
    'scales_per_octave': 3,
    'ratios': [0.5, 1, 2],
    'scales': [2**0, 2**(1.0/3), 2**(2.0/3)],
    'pyramid_levels': [3, 4, 5]

}

cfg_mobile = {
    'name': 'mobilenet0.25_retinaface_general',    # mobilenet0.25_retinageneral, mobilenet0.25_retinaface...

    # anchor
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    
    'scales_per_octave': 3,
    'ratios': [0.5, 1, 2],
    'scales': [2 ** 0, 2 ** (1.0 / 3), 2 ** (2.0 / 3)],
    'pyramid_levels': [3, 4, 5],
    # anchor

    # fpn
    'fpn': FPN,

    # train
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 500,
    'decay1': 400,
    'decay2': 450,
    'image_size': 640,
    'pretrain': False,
    'rgb_means': (104, 117, 123),
    'num_classes': 2,
    'num_workers': 4,
    'momentum': 0.9,
    'lr': 1e-3,
    'gamma': 0.1,
    'weight_decay': 5e-5,
    'resume_epoch': 0,
    'training_dataset': '/*/dataset/',
    'training_label': '/*/dataset/*/annotations.json',
    'save_weights': '/*/Synthesize/weights_detect/',
    'weights_label': '_cigarette_box',
    # train

    # net
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64,
    'pribox': PriorBox,      # PriorBoxBalance, PriorBox, GeneralAnchors
    'net': RetinaGeneral,    # RetinaFace, FaceBoxes, RetinaGeneral
    # net

    # test
    'test': cfg_test
}

cfg_re50 = {
    'name': 'resnet50_ratinaface',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,

    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': False,
    'rgb_means': (104, 117, 123),
    'num_classes': 2,
    'num_workers': 4,
    'momentum': 0.9,
    'lr': 1e-3,
    'gamma': 0.1,
    'weight_decay': 5e-4,
    'pretrain': True,
    'resume_epoch': 0,
    'training_dataset': '/*/dataset/faces/face_detect/wider_face/WIDER/WIDER_train/images/',
    'training_label': '/*/dataset/faces/face_detect/wider_face/WIDER/labels_retinaface/train/label.txt',
    'save_weights': '/*/Synthesize/weights_detect/',

    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},    # should 2, 3, 4
    'in_channel': 256,
    'out_channel': 256,
    'pribox': PriorBox,
    'net': RetinaFace,

    'test': cfg_test
}

cfg_facebox = {
    'name': 'facebox',
    'min_sizes': [[32, 64, 128], [256], [512]],  # for anchor generator
    'steps': [32, 64, 128],  # different from retina
    'variance': [0.1, 0.2],
    'clip': False,

    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'top_k': 3000,
    'keep_top_k': 400,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 1024,
    'rgb_means': (104, 117, 123),
    'num_classes': 2,
    'pretrain': False,
    'num_workers': 4,
    'momentum': 0.9,
    'lr': 1e-3,
    'gamma': 0.1,
    'weight_decay': 5e-4,
    'resume_epoch': 0,
    'training_dataset': '/*/dataset/faces/face_detect/wider_face/WIDER/WIDER_train/images/',
    'training_label': '/*/dataset/faces/face_detect/wider_face/WIDER/labels_retinaface/train/label.txt',
    'save_weights': '/*/Synthesize/weights_detect/',

    # layers
    'pribox': PriorBoxBalance,
    'net': FaceBoxes,

    'test': cfg_test
}
