cfg = {
    'arch': 'resnet',    # resnet18, efficientnet-b7
    'pretrained': False,
    'data': '',
    'save_weights': '*/Synthesize/weights_classify',
    'num_classes': 5,
    'distributed': False,
    'epochs': 22,
    'start_epoch': 0,
    'batch_size': 10,
    'workers': 0,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'momentum': 0.95,
    'print_freq': 50,    # iteration
    'resume': '*/Synthesize/weights_classify/*.pth',
    'evaluate': '',
    'image_size': (456, 456),    # 224, 600
    'datainfo': 'xxx'
}