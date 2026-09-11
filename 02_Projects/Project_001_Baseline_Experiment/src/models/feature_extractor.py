import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNetFeatureExtractor(nn.Module):

    def __init__(self):

        super().__init__()

        model = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )


        # 去掉最后分类层
        self.backbone = nn.Sequential(
            *list(model.children())[:-1]
        )


    def forward(self,x):

        x = self.backbone(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        return x