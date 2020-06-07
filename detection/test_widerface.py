from __future__ import print_function
import os
import cv2

from .configs.face_config import cfg_mobile as cfg    # cfg_res50, cfg_facebox, cfg_mobile
from .configs.detect_config import cfg_detect
from .inference_face import inference, load_model


def test():
    net = cfg['net'](cfg=cfg, phase='test')
    net = load_model(net, cfg['test']['trained_model'], False)
    net.eval()
    net = net.to('cuda')
    # print('sssssscene', net.phase)

    test_data_dir = cfg['test']['test_dataset']
    for img_dir in os.listdir(test_data_dir):
        for img_name in os.listdir(os.path.join(test_data_dir, img_dir)):
            img_path = os.path.join(test_data_dir, img_dir, img_name)
            img_raw, dets, _t = inference(cfg, net, img_path)
            save_name = os.path.join(cfg['test']['save_folder'], img_dir, img_name[:-4] + '.txt')
            dirname = os.path.dirname(save_name)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            with open(save_name, 'w') as fa:
                bboxes = dets
                file_name = os.path.basename(save_name)[:-4] + '\n'
                boxes_num = str(len(bboxes)) + '\n'
                fa.write(file_name)
                fa.write(boxes_num)
                for box in bboxes:
                    x, y = int(box[0]), int(box[1])
                    w, h = int(box[2]) - int(box[0]), int(box[3]) - int(box[1])
                    confidence = str(box[4])
                    line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                    fa.write(line)
            print('im_detect:forward_pass_time: {:.4f}s misc: {:.4f}s'.format(
                _t['forward_pass'].average_time, _t['misc'].average_time))

            if cfg['test']['save_images']:
                for b in dets:
                    if b[4] < cfg['test']['vis_thres']:
                        continue
                    text = '{:.4f}'.format(b[4])
                    b = list(map(int, b))
                    cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)    # img-->img_raw
                    cx = b[0]
                    cy = b[1] + 12
                    cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                    cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                    cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                    cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                    cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                    cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
                name = os.path.join(cfg['test']['result_folder'], img_dir, img_name)
                name_dir = os.path.dirname(name)
                if not os.path.isdir(name_dir):
                    os.makedirs(name_dir)
                cv2.imwrite(name, img_raw)


def test_one():
    pass




