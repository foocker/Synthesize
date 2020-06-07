import torch
from torch import nn
from detection.netblocks.netblock import CRelu, BasicConv2d, Inception
import torch.nn.functional as F


class FaceBoxes(nn.Module):
    def __init__(self, cfg=None, phase='train',):
        super(FaceBoxes, self).__init__()
        self.phase = phase
        self.num_classes = cfg['num_classes']     # if cfg is not None else 2
        self.conv1 = CRelu(3, 24, kernel_size=7, stride=4, padding=3)
        self.conv2 = CRelu(48, 64, kernel_size=5, stride=2, padding=2)

        self.inception1 = Inception()
        self.inception2 = Inception()
        self.inception3 = Inception()

        self.conv3_1 = BasicConv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.conv3_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv4_1 = BasicConv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.conv4_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.loc, self.conf, self.landm = self.multibox(self.num_classes)
        # print(self.loc, self.conf, self.landm)

        if self.phase == 'train':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.xavier_normal_(m.weight.data)
                        m.bias.data.fill_(0.02)
                    else:
                        m.weight.data.normal_(0, 0.01)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def multibox(self, num_classes):
        loc_layers = []
        conf_layers = []
        landm_layers = []

        loc_layers += [nn.Conv2d(128, 21 * 4, kernel_size=3, padding=1)]    # 21 = 3*4+4+4+1
        conf_layers += [nn.Conv2d(128, 21 * num_classes, kernel_size=3, padding=1)]
        landm_layers += [nn.Conv2d(128, 21 * 10, kernel_size=3, padding=1)]

        loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1) for _ in range(2)]
        conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1) for _ in range(2)]
        landm_layers += [nn.Conv2d(256, 1 * 10, kernel_size=3, padding=1) for _ in range(2)]

        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers), nn.Sequential(*landm_layers)

    def forward(self, x):
        detection_sources = []
        loc = []
        conf = []
        landm = []

        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)    # stride = 2**5
        x = self.inception1(x)    # 32*32 anchor scale
        x = self.inception2(x)    # 64*64  has change receptive field 2**6
        x = self.inception3(x)    # 128*128  2**7
        detection_sources.append(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)    # anchor: 256*256, stride=2**6, but receptive is 2**8
        detection_sources.append(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)    # 512*512, stride=2**7, recepetive is 2**9
        detection_sources.append(x)

        for (x, l1, c, l2) in zip(detection_sources, self.loc, self.conf, self.landm):
            loc.append(l1(x).permute(0, 2, 3, 1).contiguous())    # channel change to last dim
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            landm.append(l2(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)    # dim=1, concat
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        landm = torch.cat([o.view(o.size(0), -1) for o in landm], 1)

        if self.phase == 'train':
            output = (loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes),
                      landm.view(landm.size(0), -1, 10))
        else:
            output = (loc.view(loc.size(0), -1, 4), self.softmax(conf.view(-1, self.num_classes)),
                      landm.view(landm.size(0), -1, 10))
        # print(output[0].shape, output[1].shape, output[2].shape)
        # torch.Size([32, 21824, 4]) torch.Size([32, 21824, 2]) torch.Size([32, 21824, 10])
        return output
