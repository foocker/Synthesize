cfg = {
    'arch': 'efficientnet-b6',    # resnet18, efficientnet-b7
    'pretrained': False,
    'data': '',
    'save_weights': '*/Synthesize/weights_classify',
    'num_classes': 5,
    'distributed': False,
    'epochs': 40,
    'start_epoch': 0,
    'batch_size': 6,
    'workers': 0,
    'lr': 0.01,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'print_freq': 50,
    'resume': '',
    'evaluate': '',
    'image_size': (456, 456),    # 224, 600
    'datainfo': 'xxxx'
}