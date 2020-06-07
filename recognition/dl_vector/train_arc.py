from .configs.arc_config import cfg
from .representation.face_features import MobileFaceNet, Backbone
from .margin.arc_margin import Arcface
from torch.nn import CrossEntropyLoss


def train():
    if cfg.use_mobileface:
        net = MobileFaceNet(cfg.embedding_size).to(cfg.device)
    else:
        net = Backbone(cfg.net_depth, cfg.drop_ration, cfg.net_mode)

    milestones = cfg.milestones
    head = Arcface(embedding_size=cfg.embedding_size, classnum=cfg.classes_num).to(cfg.device)
    criterion = CrossEntropyLoss()
    # to do list
    # checkpoint use base ck
    # lr scheduler
    # find best lr us pip api if efficient
    # evaluate 1:N, top-k
    # find dynamic threshold strategy
    # remove pair evaluate
    # detect + multi align
    # face bank for verification
    # TensorboardX base
    # system: register + verify + statistics
