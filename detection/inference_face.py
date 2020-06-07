from __future__ import print_function
import cv2
import torch
import numpy as np

from .utils.py_cpu_nms import py_cpu_nms

from .utils.ssd_box_utils import decode, decode_landm
from utils.timer import Timer
from .configs.detect_config import cfg_detect

from detection.utils.ssd_box_utils import center_size


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def img_transform(img, **kwargs):
    if kwargs.get('origin_size') is not None:
        return img, 1
    assert 'target_size' and 'max_size' in kwargs, 'key not in kwargs'
    target_size, max_size = kwargs['target_size'], kwargs['max_size']
    img_size_min, img_size_max = np.min(img.shape[:2]), np.max(img.shape[:2])
    resize = float(target_size) / float(img_size_min)
    if np.round(resize * img_size_max) > float(img_size_max):
        resize = float(max_size) / float(img_size_max)
    img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    if 'rgb_mean' in kwargs:
        img -= kwargs['rgb_mean']
    return img, resize


def img_to_tensor(img_path, transform=None, **kwargs):
    # transform func's param from **kwargs
    img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)
    resize = 1    # original
    if transform is not None:
        img, resize = transform(img, **kwargs)
    height, width, _ = img.shape
    # sacle = torch.Tensor(img.shape[1], img.shape[0], img.shape[1], img.shape[0])
    img = img.transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img).unsqueeze(0)
    return img_raw, img_tensor, height, width, resize


def tensor_to_img(img_tensor):
    img_tensor = img_tensor.squeeze(0)
    img_np = img_tensor.numpy()
    img_np = img_np.transpose(1, 2, 0)
    return img_np


def inference(cfg, net, img_path):
    img_raw, img_tensor, height, width, resize = \
        img_to_tensor(img_path, transform=img_transform, target_size=1600, origin_size=cfg['test']['origin_size'],
                      max_size=2150, rgb_mean=cfg['rgb_means'], )    # origin_size=cfg_test['origin_size'],
    scale = torch.Tensor([width, height, width, height])
    scale = scale.to('cuda')
    img = img_tensor.to('cuda')
    _t = {'forward_pass': Timer(), 'misc': Timer()}
    _t['forward_pass'].tic()
    loc, conf, landm = net(img)
    print('loc, conf, in inference_face.py', loc, '\n', conf, )
    # print('net landm', landm, )

    _t['forward_pass'].toc()
    _t['misc'].tic()

    priorbox = cfg['pribox'](cfg, image_size=(height, width))
    # priorbox = cfg['pribox'](cfg_detect, image_size=(height, width))

    priors = priorbox.forward()
    priors = priors.to('cuda')
    # priors = center_size(priors)    # when data form is x1, y1, x2, y2
    prior_data = priors.data
    # prior_data = priors.data / height    # when generator not normal scale
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    # print(boxes.shape, prior_data.shape)
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    # print("prior_data", prior_data, prior_data.shape, cfg['variance'])
    landms = decode_landm(landm.data.squeeze(0), prior_data, cfg['variance'])
    # print('decode_landm', landms)
    scale_landm = torch.Tensor([width, height, width, height, width, height, width, height, width, height])
    scale_landm = scale_landm.to('cuda')
    landms = landms * scale_landm / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > cfg['test']['confidence_threshold'])[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1]  # or top-k
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, cfg['test']['nms_threshold'])
    dets = dets[keep, :]
    landms = landms[keep]
    # keep top-K faster NMS
    # pass
    dets = np.concatenate((dets, landms), axis=1)
    _t['misc'].toc()

    # if True:
    #     for b in dets:
    #         if b[4] < 0.4:
    #             continue
    #         text = "{:.4f}".format(b[4])
    #         b = list(map(int, b))
    #         cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
    #         cx = b[0]
    #         cy = b[1] + 12
    #         cv2.putText(img_raw, text, (cx, cy),
    #                     cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    #
    #         # landms
    #         cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
    #         cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
    #         cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
    #         cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
    #         cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
    #     # save image
    #
    #     name = "test.jpg"
    #     cv2.imwrite(name, img_raw)

    return img_raw, dets, _t

