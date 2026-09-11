import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from src.analysis.analyze_scores import analyze_scores

from src.datasets.mvtec_dataset import MVTecDataset

from src.models.feature_extractor import ResNetFeatureExtractor

from src.baseline.anomaly_baseline import AnomalyBaseline

from src.baseline.anomaly_score import AnomalyScorer

from src.evaluation.evaluate import evaluate
from src.analysis.visualize_results import visualize_samples


# =========================
# Transform
# =========================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
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



# =========================
# 1. Build Normal Center
# =========================

train_dataset = MVTecDataset(

    root="data/raw/mvtec_anomaly_detection",

    category="bottle",

    split="train",

    transform=transform

)



train_loader = DataLoader(

    train_dataset,

    batch_size=8,

    shuffle=False

)



extractor = ResNetFeatureExtractor()



baseline = AnomalyBaseline(

    extractor

)



center = baseline.fit(

    train_loader

)



print(
    "Normal center:",
    center.shape
)



# =========================
# 2. Test Dataset
# =========================

test_dataset = MVTecDataset(

    root="data/raw/mvtec_anomaly_detection",

    category="bottle",

    split="test",

    transform=transform

)



test_loader = DataLoader(

    test_dataset,

    batch_size=8,

    shuffle=False

)



scorer = AnomalyScorer(

    center

)



all_scores = []

all_labels = []

all_paths = []



# =========================
# 3. Calculate Scores
# =========================

extractor.eval()


with torch.no_grad():


    for images, labels, paths in test_loader:


        features = extractor(
            images
        )


        scores = scorer.score(
            features
        )


        all_scores.extend(
            scores.cpu().numpy()
        )


        all_labels.extend(
            labels.cpu().numpy()
        )


        all_paths.extend(
            paths
        )



# =========================
# 4. Evaluation
# =========================

auc = evaluate(

    all_scores,

    all_labels

)



print("====================")

print("Experiment Result")

print("====================")


print(
    "Category: bottle"
)


print(
    "Images:",
    len(all_scores)
)


print(
    "AUROC:",
    auc
)



# =========================
# 5. Failure Analysis Preview
# =========================

print("====================")

print("Top anomaly samples")

print("====================")


results = list(
    zip(
        all_scores,
        all_labels,
        all_paths
    )
)



results.sort(
    key=lambda x:x[0],
    reverse=True
)



for score, label, path in results[:5]:

    print("--------------------")

    print(
        "Score:",
        score
    )

    print(
        "Label:",
        label
    )

    print(
        "Path:",
        path
    )
analysis = analyze_scores(

    all_scores,

    all_labels,

    all_paths

)
visualize_samples(

    analysis["top_anomaly"],

    "results/top_anomaly.png",

    "Top Anomaly"

)



visualize_samples(

    analysis["hard_defect"],

    "results/hard_defect.png",

    "Hard Defect"

)



visualize_samples(

    analysis["false_positive"],

    "results/false_positive.png",

    "False Positive"

)

print("====================")
print("Hard Defect")
print("====================")


for score,label,path in analysis["hard_defect"]:

    print("--------------------")

    print("Score:",score)

    print("Label:",label)

    print("Path:",path)



print("====================")
print("False Positive")
print("====================")


for score,label,path in analysis["false_positive"]:

    print("--------------------")

    print("Score:",score)

    print("Label:",label)

    print("Path:",path)