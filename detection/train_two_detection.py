from torch.optim import SGD
from ..checkpoint import Checkpointer
from ..data.coco_vison import get_data_loader
from torch.optim.lr_scheduler import MultiStepLR
from .models.fasterrcnn import fasterrcnn_resnetxx_fpnxx
from .configs.config_two_stage import cfg_two_stage, Struct_Component_Cfg


def train_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler=None, management=True):
    for i, (imgs, targets) in enumerate(data_loader):
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        print(f'Iteration: {i}/{len(data_loader)}, Loss: {losses}')
    if (epoch+1) % 10 == 0 and management:
        # assert ckp is not None, 'Checkpointer is None'
        ckp = Checkpointer(model, save_dir='/vdata/Synthesize/weights_detect/')
        ckp.save(name='{}_{}_epoch_{}_loss_{:.4f}'.format('resnet18', 'voc07_to_coco', epoch, losses.cpu().item()))
        print('saved!!!!!', '{}_epoch_{}_loss_{:.4f}'.format('voc07_to_coco', epoch, losses.cpu().item()))


def train(epochs=100):
    model = fasterrcnn_resnetxx_fpnxx(Struct_Component_Cfg)
    device = 'cuda:0'
    model = model.to(device)
    model.train()
    optimizer = SGD(model.parameters(), lr=8e-3, momentum=0.9)
    lr_scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
    data_loader = get_data_loader(**cfg_two_stage)
    # ckp = Checkpointer(model, save_dir='/vdata/Synthesize/weights_detect/')
    for ep in range(epochs):
        train_epoch(model, optimizer, data_loader, device, ep, lr_scheduler=lr_scheduler, management=True)

