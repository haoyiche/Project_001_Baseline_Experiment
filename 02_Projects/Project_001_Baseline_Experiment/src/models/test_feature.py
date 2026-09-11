import torch

from feature_extractor import ResNetFeatureExtractor


model = ResNetFeatureExtractor()


x = torch.randn(
    8,
    3,
    224,
    224
)


feature = model(x)


print(
    "Input:",
    x.shape
)


print(
    "Feature:",
    feature.shape
)