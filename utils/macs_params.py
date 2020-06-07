from torchvision.models import resnet18
from thop import profile, clever_format
import torch


model = resnet18()
input_ = torch.randn(1, 3, 224, 224)
macs, params = profile(model, inputs=(input_, ))
print(macs, params)
macs, params = clever_format([macs, params], "%.3f")
print(macs, params)

