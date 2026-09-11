from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.mvtec_dataset import MVTecDataset
from src.models.feature_extractor import ResNetFeatureExtractor
from src.baseline.anomaly_baseline import AnomalyBaseline



transform = transforms.Compose([

    transforms.Resize(
        (224,224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )

])



dataset = MVTecDataset(
    "data/raw/mvtec_anomaly_detection",
    "bottle",
    "train",
    transform
)


loader = DataLoader(
    dataset,
    batch_size=8
)



extractor = ResNetFeatureExtractor()



baseline = AnomalyBaseline(
    extractor
)


center = baseline.fit(loader)


print(
    "Center shape:",
    center.shape
)