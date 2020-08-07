# model.py
import torch.nn as nn
import pretrainedmodels
import efficientnet_pytorch
from torch.nn import functional as F


class SEResNext50_32x4d(nn.Module):
    def __init__(self, pretrained):
        super(SEResNext50_32x4d, self).__init__()
        self.model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"
        ](pretrained=pretrained)
        self.out = nn.Linear(2048, 1)

    def forward(self, image, targets):
        bs, _, _, _ = image.shape
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)
        out = self.out(x)
        loss = nn.BCEWithLogitsLoss()(
            out, targets.reshape(-1, 1).type_as(out)
        ).cuda(gpu)
        return out, loss


class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        self.base_model = efficientnet_pytorch.EfficientNet.from_pretrained(
            'efficientnet-b7'
        )
        self.base_model._fc = nn.Linear(
            in_features=2560,
            out_features=1,
            bias=True
        )

    def forward(self, image, targets):
        out = self.base_model(image)
        loss = nn.BCEWithLogitsLoss()(out, targets.view(-1, 1).type_as(out))
        return out, loss

