from torch.utils.data import DataLoader
from torchvision import transforms
import torch

from src.datasets.mvtec_dataset import MVTecDataset
from src.models.feature_extractor import ResNetFeatureExtractor
from src.baseline.anomaly_baseline import AnomalyBaseline
from src.baseline.anomaly_score import AnomalyScorer



transform = transforms.Compose([

    transforms.Resize(
        (224,224)
    ),


    transforms.ToTensor(),


    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )

])



# ======================
# 1. 建立正常中心
# ======================


train_dataset = MVTecDataset(
    "data/raw/mvtec_anomaly_detection",
    "bottle",
    "train",
    transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=8
)



extractor = ResNetFeatureExtractor()



baseline = AnomalyBaseline(
    extractor
)



center = baseline.fit(
    train_loader
)



# ======================
# 2. 测试
# ======================


test_dataset = MVTecDataset(
    "data/raw/mvtec_anomaly_detection",
    "bottle",
    "test",
    transform
)


test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=True
)



scorer = AnomalyScorer(
    center
)



extractor.eval()


with torch.no_grad():

    for images, labels in test_loader:


        features = extractor(
            images
        )


        scores = scorer.score(
            features
        )


        print(
            "Scores:"
        )

        print(
            scores
        )


        print(
            "Labels:"
        )

        print(
            labels
        )


        break