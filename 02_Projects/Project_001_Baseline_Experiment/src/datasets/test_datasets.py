from torchvision import transforms
from mvtec_dataset import MVTecDataset


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


dataset = MVTecDataset(
    "data/raw/mvtec_anomaly_detection",
    "bottle",
    "train",
    transform
)


print(
    "Dataset size:",
    len(dataset)
)


image = dataset[0]


print(
    "Tensor shape:",
    image.shape
)


print(
    "dtype:",
    image.dtype
)


print(
    "min:",
    image.min()
)


print(
    "max:",
    image.max()
)