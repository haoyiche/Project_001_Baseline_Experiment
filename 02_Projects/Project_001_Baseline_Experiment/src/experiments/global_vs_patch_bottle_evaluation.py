from pathlib import Path
import csv

from PIL import Image

import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
)

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)


# ============================================================
# Dataset
# ============================================================

class ImageDataset(Dataset):

    def __init__(
        self,
        image_paths,
        preprocess,
    ):
        self.image_paths = list(
            image_paths
        )

        self.preprocess = preprocess

    def __len__(self):
        return len(
            self.image_paths
        )

    def __getitem__(
        self,
        index,
    ):
        path = self.image_paths[
            index
        ]

        image = Image.open(
            path
        ).convert("RGB")

        image = self.preprocess(
            image
        )

        return image


# ============================================================
# Global Feature Extraction
# ============================================================

@torch.inference_mode()
def extract_global_features(
    model,
    image_paths,
    preprocess,
    device,
    batch_size=16,
):
    """
    ResNet18 with fc = Identity:

    input
        [B,3,224,224]

    output
        [B,512]

    The output is the global pooled
    layer4 representation.
    """

    dataset = ImageDataset(
        image_paths=image_paths,
        preprocess=preprocess,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    feature_parts = []

    for images in loader:

        images = images.to(
            device
        )

        features = model(
            images
        )

        feature_parts.append(
            features.cpu()
        )

    return torch.cat(
        feature_parts,
        dim=0,
    )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    labels,
    scores,
):
    auroc = roc_auc_score(
        labels,
        scores,
    )

    ap = average_precision_score(
        labels,
        scores,
    )

    return {
        "auroc": float(auroc),
        "ap": float(ap),
    }


def summarize_method(
    name,
    rows,
    score_key,
):

    labels = [
        row["label"]
        for row in rows
    ]

    scores = [
        row[score_key]
        for row in rows
    ]

    metrics = calculate_metrics(
        labels,
        scores,
    )

    good_scores = torch.tensor(
        [
            row[score_key]
            for row in rows
            if row["label"] == 0
        ],
        dtype=torch.float32,
    )

    defect_scores = torch.tensor(
        [
            row[score_key]
            for row in rows
            if row["label"] == 1
        ],
        dtype=torch.float32,
    )

    summary = {
        "auroc":
            metrics["auroc"],

        "ap":
            metrics["ap"],

        "good_mean":
            good_scores.mean().item(),

        "good_p95":
            torch.quantile(
                good_scores,
                0.95,
            ).item(),

        "good_max":
            good_scores.max().item(),

        "defect_min":
            defect_scores.min().item(),

        "defect_mean":
            defect_scores.mean().item(),

        "defect_max":
            defect_scores.max().item(),

        "margin":
            (
                defect_scores.min()
                - good_scores.max()
            ).item(),
    }

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"\nOverall AUROC: "
        f"{summary['auroc']:.6f}"
    )

    print(
        f"Average Precision: "
        f"{summary['ap']:.6f}"
    )

    print("\nGOOD:")

    print(
        f"  mean = "
        f"{summary['good_mean']:.6f}"
    )

    print(
        f"  P95  = "
        f"{summary['good_p95']:.6f}"
    )

    print(
        f"  max  = "
        f"{summary['good_max']:.6f}"
    )

    print("\nALL DEFECTS:")

    print(
        f"  min  = "
        f"{summary['defect_min']:.6f}"
    )

    print(
        f"  mean = "
        f"{summary['defect_mean']:.6f}"
    )

    print(
        f"  max  = "
        f"{summary['defect_max']:.6f}"
    )

    print(
        f"\nSeparation margin: "
        f"{summary['margin']:.6f}"
    )

    return summary


# ============================================================
# Per Defect Evaluation
# ============================================================

def evaluate_per_defect(
    method_name,
    rows,
    score_key,
):

    defect_types = [
        "broken_large",
        "broken_small",
        "contamination",
    ]

    results = {}

    print("\n" + "-" * 70)

    print(
        f"{method_name} - Per Defect Type"
    )

    print("-" * 70)

    for defect_type in defect_types:

        subset = [
            row
            for row in rows
            if (
                row["class"] == "good"
                or
                row["class"]
                == defect_type
            )
        ]

        labels = [
            0
            if row["class"] == "good"
            else 1
            for row in subset
        ]

        scores = [
            row[score_key]
            for row in subset
        ]

        metrics = calculate_metrics(
            labels,
            scores,
        )

        results[
            defect_type
        ] = metrics

        print(
            f"\n{defect_type}:"
        )

        print(
            f"  AUROC = "
            f"{metrics['auroc']:.6f}"
        )

        print(
            f"  AP    = "
            f"{metrics['ap']:.6f}"
        )

    return results


# ============================================================
# Failure Candidates
# ============================================================

def print_failure_candidates(
    method_name,
    rows,
    score_key,
):

    good_rows = [
        row
        for row in rows
        if row["label"] == 0
    ]

    defect_rows = [
        row
        for row in rows
        if row["label"] == 1
    ]

    highest_good = sorted(
        good_rows,
        key=lambda row:
            row[score_key],
        reverse=True,
    )[:5]

    lowest_defect = sorted(
        defect_rows,
        key=lambda row:
            row[score_key],
    )[:5]

    print("\n" + "=" * 70)

    print(
        f"{method_name} - Failure Candidates"
    )

    print("=" * 70)

    print(
        "\nTop-5 highest-scoring GOOD:"
    )

    for rank, row in enumerate(
        highest_good,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{row['relative_path']} "
            f"score="
            f"{row[score_key]:.6f}"
        )

    print(
        "\nTop-5 lowest-scoring DEFECT:"
    )

    for rank, row in enumerate(
        lowest_defect,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{row['relative_path']} "
            f"score="
            f"{row[score_key]:.6f}"
        )

    return {
        "highest_good":
            highest_good,

        "lowest_defect":
            lowest_defect,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)

    print(
        "EXP017 - Global vs Patch Bottle Evaluation"
    )

    print("=" * 70)

    # ========================================================
    # 1. Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    # ========================================================
    # 2. Load Patch EXP016 CSV
    # ========================================================

    patch_csv_path = Path(
        "results/exp016/"
        "full_bottle_scores.csv"
    )

    if not patch_csv_path.exists():
        raise FileNotFoundError(
            f"EXP016 CSV not found:\n"
            f"{patch_csv_path.resolve()}"
        )

    patch_rows = []

    with open(
        patch_csv_path,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:

            # Normalize Windows path separator.
            relative_path = (
                row["relative_path"]
                .replace("\\", "/")
            )

            patch_rows.append(
                {
                    "relative_path":
                        relative_path,

                    "class":
                        row["class"],

                    "label":
                        int(row["label"]),

                    "patch_score":
                        float(
                            row["coreset100"]
                        ),
                }
            )

    print(
        f"\nLoaded EXP016 rows: "
        f"{len(patch_rows)}"
    )

    if len(patch_rows) != 83:
        raise RuntimeError(
            "Expected 83 Bottle test images."
        )

    # ========================================================
    # 3. Model
    # ========================================================

    weights = (
        ResNet18_Weights.DEFAULT
    )

    preprocess = (
        weights.transforms()
    )

    model = resnet18(
        weights=weights
    )

    # Replace classifier with identity.
    #
    # Standard ResNet18 forward:
    #
    # layer4
    # -> avgpool
    # -> flatten
    # -> fc
    #
    # After fc = Identity:
    #
    # output = [B,512]
    #
    # This is our global image feature.

    model.fc = torch.nn.Identity()

    model.eval()
    model.to(device)

    # ========================================================
    # 4. Build Global Normal Center
    # ========================================================

    train_good_dir = Path(
        "data/raw/mvtec_anomaly_detection/"
        "bottle/train/good"
    )

    train_paths = sorted(
        train_good_dir.glob(
            "*.png"
        )
    )

    print(
        f"\nTrain normal images: "
        f"{len(train_paths)}"
    )

    train_features = (
        extract_global_features(
            model=model,
            image_paths=train_paths,
            preprocess=preprocess,
            device=device,
        )
    )

    print(
        "\nTrain global features:"
    )

    print(
        train_features.shape
    )

    normal_center = (
        train_features.mean(
            dim=0
        )
    )

    print(
        "\nNormal center:"
    )

    print(
        normal_center.shape
    )

    # ========================================================
    # 5. Exact Same Test Set as EXP016
    # ========================================================

    test_root = Path(
        "data/raw/mvtec_anomaly_detection/"
        "bottle/test"
    )

    test_paths = [
        test_root
        / Path(
            row["relative_path"]
        )
        for row in patch_rows
    ]

    for path in test_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Test image not found:\n"
                f"{path}"
            )

    # ========================================================
    # 6. Extract Global Test Features
    # ========================================================

    test_features = (
        extract_global_features(
            model=model,
            image_paths=test_paths,
            preprocess=preprocess,
            device=device,
        )
    )

    print(
        "\nTest global features:"
    )

    print(
        test_features.shape
    )

    # ========================================================
    # 7. Global Anomaly Score
    # ========================================================

    global_scores = torch.linalg.vector_norm(
        test_features
        - normal_center.unsqueeze(0),
        ord=2,
        dim=1,
    )

    print(
        "\nGlobal scores:"
    )

    print(
        global_scores.shape
    )

    # ========================================================
    # 8. Merge Global + Patch
    # ========================================================

    rows = []

    for index, patch_row in enumerate(
        patch_rows
    ):

        rows.append(
            {
                "relative_path":
                    patch_row[
                        "relative_path"
                    ],

                "class":
                    patch_row[
                        "class"
                    ],

                "label":
                    patch_row[
                        "label"
                    ],

                "global_score":
                    global_scores[
                        index
                    ].item(),

                "patch_score":
                    patch_row[
                        "patch_score"
                    ],
            }
        )

    # ========================================================
    # 9. Overall Metrics
    # ========================================================

    global_summary = (
        summarize_method(
            name="Global Baseline",
            rows=rows,
            score_key="global_score",
        )
    )

    patch_summary = (
        summarize_method(
            name="Patch Coreset100",
            rows=rows,
            score_key="patch_score",
        )
    )

    # ========================================================
    # 10. Per Defect Type
    # ========================================================

    global_per_defect = (
        evaluate_per_defect(
            method_name=
                "Global Baseline",

            rows=rows,

            score_key=
                "global_score",
        )
    )

    patch_per_defect = (
        evaluate_per_defect(
            method_name=
                "Patch Coreset100",

            rows=rows,

            score_key=
                "patch_score",
        )
    )

    # ========================================================
    # 11. Failure Candidates
    # ========================================================

    global_failures = (
        print_failure_candidates(
            method_name=
                "Global Baseline",

            rows=rows,

            score_key=
                "global_score",
        )
    )

    patch_failures = (
        print_failure_candidates(
            method_name=
                "Patch Coreset100",

            rows=rows,

            score_key=
                "patch_score",
        )
    )

    # ========================================================
    # 12. Direct Metric Comparison
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "Global vs Patch Summary"
    )

    print("=" * 70)

    print(
        "\nOverall AUROC:"
    )

    print(
        f"  Global = "
        f"{global_summary['auroc']:.6f}"
    )

    print(
        f"  Patch  = "
        f"{patch_summary['auroc']:.6f}"
    )

    print(
        "\nAverage Precision:"
    )

    print(
        f"  Global = "
        f"{global_summary['ap']:.6f}"
    )

    print(
        f"  Patch  = "
        f"{patch_summary['ap']:.6f}"
    )

    print(
        "\nPer-defect AUROC:"
    )

    for defect_type in [
        "broken_large",
        "broken_small",
        "contamination",
    ]:

        print(
            f"\n{defect_type}:"
        )

        print(
            f"  Global = "
            f"{global_per_defect[defect_type]['auroc']:.6f}"
        )

        print(
            f"  Patch  = "
            f"{patch_per_defect[defect_type]['auroc']:.6f}"
        )

    # ========================================================
    # 13. Save CSV
    # ========================================================

    output_dir = Path(
        "results/exp017"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "global_vs_patch_scores.csv"
    )

    with open(
        csv_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "relative_path",
                "class",
                "label",
                "global_score",
                "patch_score",
            ],
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # ========================================================
    # 14. Save Summary
    # ========================================================

    summary_path = (
        output_dir
        / "global_vs_patch_summary.pt"
    )

    torch.save(
        {
            "global_summary":
                global_summary,

            "patch_summary":
                patch_summary,

            "global_per_defect":
                global_per_defect,

            "patch_per_defect":
                patch_per_defect,

            "global_failures":
                global_failures,

            "patch_failures":
                patch_failures,
        },
        summary_path,
    )

    print("\n" + "=" * 70)
    print("Saved Results")
    print("=" * 70)

    print("\nCSV:")
    print(
        csv_path.resolve()
    )

    print("\nSummary:")
    print(
        summary_path.resolve()
    )

    print("\n" + "=" * 70)
    print("Experiment completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()