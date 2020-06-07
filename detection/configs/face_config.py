from ..anchors import PriorBoxBalance, PriorBox, GeneralAnchors
from ..models import RetinaFace, FaceBoxes, RetinaGeneral
from ..models.retinageneral import FPN, FPNGeneral


cfg_test = {
    'confidence_threshold': 0.02,
    'top_k': 5000,
    'nms_threshold': 0.4,
    'keep_top_k': 750,
    'save_images': True,
    'vis_thres': 0.5,  # visualization_threshold
    'origin_size': True,  # Whether use origin image size to evaluate, mobilenet0.25_retinaface_general_epoch_5.pth
    'trained_model': '*/Synthesize/weights_detect/mobilenet0.25_Final_Retinaface.pth',  # mobilenet0.25_retinaface_general_Final.pth
    'save_folder': '*/dataset/faces/face_detect/wider_face/WIDER/WIDER_val/Pred_txt/',
    'result_folder': '*/dataset/faces/face_detect/wider_face/WIDER/WIDER_val/result/',

    # 'trained_model': '*/Synthesize/weights_detect/facebox_Final_.pth',
    # 'confidence_threshold': 0.02,
    # 'top_k': 3000,
    # 'nms_threshold': 0.4,
    # 'keep_top_k': 400,
    # 'save_images': True,
    # 'vis_thres': 0.5,  # visualization_threshold
    # 'origin_size': True,  # Whether use origin image size to evaluate
    # 'save_folder': '',
    # 'result_folder': '',

    'test_dataset': '',
    'test_label': '*/dataset/faces/face_detect/wider_face/WIDER/labels_retinaface/val/label.txt',

}


cfg_mobile = {
    'name': 'mobilenet0.25_retinaface',    # mobilenet0.25_retinageneral, ...

    # anchor
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,

    'ratios': [0.5, 1, 2],
    'scales': [2 ** 0, 2 ** (1.0 / 3), 2 ** (2.0 / 3)],
    'pyramid_levels': [3, 4, 5],
    # anchor

    # fpn
    'fpn': FPN,    # FPNGeneral, FPN

    # train
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': False,
    'rgb_means': (104, 117, 123),
    'num_classes': 2,
    'num_workers': 4,
    'momentum': 0.9,
    'lr': 1e-3,
    'gamma': 0.1,
    'weight_decay': 5e-4,
    'resume_epoch': 0,
    'training_dataset': '',
    'training_label': '*/dataset/faces/face_detect/wider_face/WIDER/labels_retinaface/train/label.txt',
    'save_weights': '*/Synthesize/weights_detect/',
    # train

    # net
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64,
    'pribox': PriorBox,      # PriorBoxBalance, PriorBox, GeneralAnchors
    'net': RetinaFace,    # RetinaFace, FaceBoxes, RetinaGeneral
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
    'training_dataset': '*/dataset/faces/face_detect/wider_face/WIDER/WIDER_train/images/',
    'training_label': '*/dataset/faces/face_detect/wider_face/WIDER/labels_retinaface/train/label.txt',
    'save_weights': '*/Synthesize/weights_detect/',

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
    'training_dataset': '*/dataset/faces/face_detect/wider_face/WIDER/WIDER_train/images/',
    'training_label': '*/dataset/faces/face_detect/wider_face/WIDER/labels_retinaface/train/label.txt',
    'save_weights': '*/Synthesize/weights_detect/',

    # layers
    'pribox': PriorBoxBalance,
    'net': FaceBoxes,

    'test': cfg_test
}
