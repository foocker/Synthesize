EFFICIENTDET = {
    'efficientdet-d0': {'input_size': 512,
                        'backbone': 'B0',
                        'W_bifpn': 64,
                        'D_bifpn': 2,
                        'D_class': 3},
    'efficientdet-d1': {'input_size': 640,
                        'backbone': 'B1',
                        'W_bifpn': 88,
                        'D_bifpn': 3,
                        'D_class': 3},
    'efficientdet-d2': {'input_size': 768,
                        'backbone': 'B2',
                        'W_bifpn': 112,
                        'D_bifpn': 4,
                        'D_class': 3},
    'efficientdet-d3': {'input_size': 896,
                        'backbone': 'B3',
                        'W_bifpn': 160,
                        'D_bifpn': 5,
                        'D_class': 4},
    'efficientdet-d4': {'input_size': 1024,
                        'backbone': 'B4',
                        'W_bifpn': 224,
                        'D_bifpn': 6,
                        'D_class': 4},
    'efficientdet-d5': {'input_size': 1280,
                        'backbone': 'B5',
                        'W_bifpn': 288,
                        'D_bifpn': 7,
                        'D_class': 4},
    'efficientdet-d6': {'input_size': 1408,
                        'backbone': 'B6',
                        'W_bifpn': 384,
                        'D_bifpn': 8,
                        'D_class': 5},
    'efficientdet-d7': {'input_size': 1636,
                        'backbone': 'B6',
                        'W_bifpn': 384,
                        'D_bifpn': 8,
                        'D_class': 5},
}


cfg_retinahead = {
    'feat_channels': 256,
    'anchor_ratios': [8, 16, 32],
    'anchor_scales': [0.5, 1.0, 2.0],
    'conv_cfg': None,    # in Conv_Module
    'norm_cfg': None,
    'stacked_convs': 4,
}


cfg_anchor = {
    'anchor_sizes': '',
    'aspect_ratios': '',
    'anchor_strides': '',
    'straddle_thresh': '',
    'octave': '',
    'scales_per_octave': 3,
    'ratios': [0.5, 1, 2],
    'scales': [2 ** 0, 2 ** (1.0 / 3), 2 ** (2.0 / 3)],
    'pyramid_levels': [3, 4, 5, 6, 7]
}


class Args(object):
    # for train, evaluate
    def __init__(self):
        self.resume = ''
        self.dataset = 'COCO'
        self.data_root = '/aidata/dataset/coco'
        self.network = 'efficientdet-d1'
        self.resume = ''    # path
        self.num_epoch = 100
        self.batch_size = 6
        self.num_worker = 4
        self.num_classes = 80
        self.device = [0]
        self.grad_accumulation_steps = 1
        self.lr = 1e-4
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.gamma = 0.1
        self.save_folder = '/vdata/Synthesize/weights_detect/'
        self.save_freq_ep = 5