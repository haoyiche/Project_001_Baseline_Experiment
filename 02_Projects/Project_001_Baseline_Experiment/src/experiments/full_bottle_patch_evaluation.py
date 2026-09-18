from pathlib import Path
import csv

import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)

from src.experiments.single_patch_nearest_neighbor import (
    build_memory_bank,
)

from src.experiments.full_image_patch_scoring import (
    extract_image_patches,
)


def score_image(
    query_patches: torch.Tensor,
    memory_bank: torch.Tensor,
):
    """
    Exact patch-level nearest-neighbor scoring.

    query_patches:
        [784, 128]

    memory_bank:
        [M, 128]

    Returns:
        image_score:
            scalar

        patch_scores:
            [784]
    """

    distances = torch.cdist(
        query_patches,
        memory_bank,
        p=2,
    )

    patch_scores = (
        distances.min(dim=1).values
    )

    image_score = (
        patch_scores.max().item()
    )

    return (
        image_score,
        patch_scores,
    )


def calculate_binary_metrics(
    labels,
    scores,
):
    """
    label:
        0 = good
        1 = anomaly
    """

    auroc = roc_auc_score(
        labels,
        scores,
    )

    average_precision = (
        average_precision_score(
            labels,
            scores,
        )
    )

    return {
        "auroc": float(auroc),
        "ap": float(average_precision),
    }


def summarize_overall(
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

    metrics = calculate_binary_metrics(
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

        "defect_mean":
            defect_scores.mean().item(),

        "defect_min":
            defect_scores.min().item(),

        "defect_max":
            defect_scores.max().item(),

        "separation_margin":
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
        f"{summary['separation_margin']:.6f}"
    )

    return summary


def evaluate_defect_type(
    rows,
    score_key,
    defect_type,
):
    """
    Evaluate:
        good
        vs
        one defect type
    """

    subset = [
        row
        for row in rows
        if (
            row["class"] == "good"
            or
            row["class"] == defect_type
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

    return calculate_binary_metrics(
        labels,
        scores,
    )


def print_per_defect_metrics(
    method_name,
    rows,
    score_key,
):
    defect_types = [
        "broken_large",
        "broken_small",
        "contamination",
    ]

    result = {}

    print("\n" + "-" * 70)

    print(
        f"{method_name} - Per Defect Type"
    )

    print("-" * 70)

    for defect_type in defect_types:

        metrics = evaluate_defect_type(
            rows=rows,
            score_key=score_key,
            defect_type=defect_type,
        )

        result[defect_type] = metrics

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

    return result


def print_failure_candidates(
    rows,
    score_key,
    method_name,
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
        key=lambda x: x[score_key],
        reverse=True,
    )[:5]

    lowest_defect = sorted(
        defect_rows,
        key=lambda x: x[score_key],
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


def main():

    print("=" * 70)

    print(
        "EXP016 - Full Bottle Patch Evaluation"
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
    # 2. Model
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

    model.eval()
    model.to(device)

    # ========================================================
    # 3. Build Full Normal Memory Bank
    # ========================================================

    train_good_dir = Path(
        "data/raw/mvtec_anomaly_detection/"
        "bottle/train/good"
    )

    print(
        "\nBuilding full normal memory bank..."
    )

    full_memory_bank = (
        build_memory_bank(
            model=model,
            preprocess=preprocess,
            train_good_dir=train_good_dir,
            device=device,
        )
    )

    print("\nFull memory bank:")
    print(full_memory_bank.shape)

    # ========================================================
    # 4. Reconstruct EXP014 Memory Banks
    # ========================================================

    exp014_path = Path(
        "results/exp014/"
        "minimal_coreset_experiment.pt"
    )

    exp014 = torch.load(
        exp014_path,
        map_location="cpu",
    )

    candidate_indices = (
        exp014["candidate_indices"]
    )

    random_indices = (
        exp014["random_indices"]
    )

    coreset_indices = (
        exp014["coreset_indices"]
    )

    candidate_bank = (
        full_memory_bank[
            candidate_indices
        ]
    )

    random_bank = (
        candidate_bank[
            random_indices
        ]
    )

    coreset_bank = (
        candidate_bank[
            coreset_indices
        ]
    )

    print("\nMemory Banks:")

    print(
        f"Candidate5000: "
        f"{tuple(candidate_bank.shape)}"
    )

    print(
        f"Random100:     "
        f"{tuple(random_bank.shape)}"
    )

    print(
        f"Coreset100:    "
        f"{tuple(coreset_bank.shape)}"
    )

    # ========================================================
    # 5. Collect ALL Bottle Test Images
    # ========================================================

    test_root = Path(
        "data/raw/mvtec_anomaly_detection/"
        "bottle/test"
    )

    class_names = [
        "good",
        "broken_large",
        "broken_small",
        "contamination",
    ]

    samples = []

    for class_name in class_names:

        class_dir = (
            test_root
            / class_name
        )

        paths = sorted(
            class_dir.glob("*.png")
        )

        print(
            f"\n{class_name}: "
            f"{len(paths)} images"
        )

        for path in paths:

            label = (
                0
                if class_name == "good"
                else 1
            )

            samples.append(
                {
                    "path": path,
                    "class": class_name,
                    "label": label,
                }
            )

    print(
        f"\nTotal test images: "
        f"{len(samples)}"
    )

    # ========================================================
    # 6. Evaluate
    # ========================================================

    rows = []

    for index, sample in enumerate(
        samples,
        start=1,
    ):

        image_path = sample["path"]

        class_name = sample["class"]

        label = sample["label"]

        relative_path = (
            image_path.relative_to(
                test_root
            )
        )

        print(
            f"\n[{index:02d}/"
            f"{len(samples):02d}] "
            f"{relative_path}"
        )

        (
            query_patches,
            _,
            _,
        ) = extract_image_patches(
            image_path=image_path,
            model=model,
            preprocess=preprocess,
            device=device,
        )

        (
            candidate_score,
            _,
        ) = score_image(
            query_patches,
            candidate_bank,
        )

        (
            random_score,
            _,
        ) = score_image(
            query_patches,
            random_bank,
        )

        (
            coreset_score,
            _,
        ) = score_image(
            query_patches,
            coreset_bank,
        )

        print(
            f"  Candidate5000 = "
            f"{candidate_score:.6f}"
        )

        print(
            f"  Random100     = "
            f"{random_score:.6f}"
        )

        print(
            f"  Coreset100    = "
            f"{coreset_score:.6f}"
        )

        rows.append(
            {
                "relative_path":
                    str(relative_path),

                "class":
                    class_name,

                "label":
                    label,

                "candidate5000":
                    candidate_score,

                "random100":
                    random_score,

                "coreset100":
                    coreset_score,
            }
        )

    # ========================================================
    # 7. Overall Summaries
    # ========================================================

    candidate_summary = summarize_overall(
        name="Candidate5000",
        rows=rows,
        score_key="candidate5000",
    )

    random_summary = summarize_overall(
        name="Random100",
        rows=rows,
        score_key="random100",
    )

    coreset_summary = summarize_overall(
        name="Coreset100",
        rows=rows,
        score_key="coreset100",
    )

    # ========================================================
    # 8. Per-Defect Metrics
    # ========================================================

    candidate_per_defect = (
        print_per_defect_metrics(
            method_name="Candidate5000",
            rows=rows,
            score_key="candidate5000",
        )
    )

    random_per_defect = (
        print_per_defect_metrics(
            method_name="Random100",
            rows=rows,
            score_key="random100",
        )
    )

    coreset_per_defect = (
        print_per_defect_metrics(
            method_name="Coreset100",
            rows=rows,
            score_key="coreset100",
        )
    )

    # ========================================================
    # 9. Failure Candidates
    # ========================================================

    candidate_failures = (
        print_failure_candidates(
            rows=rows,
            score_key="candidate5000",
            method_name="Candidate5000",
        )
    )

    coreset_failures = (
        print_failure_candidates(
            rows=rows,
            score_key="coreset100",
            method_name="Coreset100",
        )
    )

    # ========================================================
    # 10. Save Results
    # ========================================================

    output_dir = Path(
        "results/exp016"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "full_bottle_scores.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "relative_path",
                "class",
                "label",
                "candidate5000",
                "random100",
                "coreset100",
            ],
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    summary_path = (
        output_dir
        / "full_bottle_summary.pt"
    )

    torch.save(
        {
            "candidate_summary":
                candidate_summary,

            "random_summary":
                random_summary,

            "coreset_summary":
                coreset_summary,

            "candidate_per_defect":
                candidate_per_defect,

            "random_per_defect":
                random_per_defect,

            "coreset_per_defect":
                coreset_per_defect,

            "candidate_failures":
                candidate_failures,

            "coreset_failures":
                coreset_failures,
        },
        summary_path,
    )

    print("\n" + "=" * 70)
    print("Saved Results")
    print("=" * 70)

    print("\nCSV:")
    print(csv_path.resolve())

    print("\nSummary:")
    print(summary_path.resolve())

    print("\n" + "=" * 70)
    print("Experiment completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()