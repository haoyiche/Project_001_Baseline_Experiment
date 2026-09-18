from pathlib import Path
import csv

import torch
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)

from src.experiments.coreset_stability_repeated_seed import (
    extract_patch_features_for_paths,
    build_full_memory,
    build_coreset,
    binary_metrics,
    per_defect_metrics,
)

from src.experiments.threshold_calibration_normal_only import (
    confusion_at_threshold,
)


# ============================================================
# Patch Score Extraction
# ============================================================

@torch.inference_mode()
def compute_patch_score_vectors(
    image_features,
    memory_bank,
):
    """
    image_features:
        list of tensors,
        each [784,128]

    return:
        list of tensors,
        each [784]

    IMPORTANT:
    NN distances are computed only once.
    Different image-level aggregation methods
    reuse exactly the same patch scores.
    """

    all_patch_scores = []

    for patches in image_features:

        distances = torch.cdist(
            patches,
            memory_bank,
            p=2,
        )

        patch_scores = distances.min(
            dim=1
        ).values

        all_patch_scores.append(
            patch_scores.cpu()
        )

    return all_patch_scores


# ============================================================
# Aggregation
# ============================================================

def aggregate_patch_scores(
    patch_scores,
    method,
):

    if method == "max":

        return patch_scores.max().item()

    if method == "top5_mean":

        return (
            torch.topk(
                patch_scores,
                k=5,
            )
            .values
            .mean()
            .item()
        )

    if method == "top20_mean":

        return (
            torch.topk(
                patch_scores,
                k=20,
            )
            .values
            .mean()
            .item()
        )

    if method == "mean":

        return (
            patch_scores
            .mean()
            .item()
        )

    raise ValueError(
        f"Unknown aggregation method: "
        f"{method}"
    )


def aggregate_dataset(
    patch_score_vectors,
    method,
):

    scores = [
        aggregate_patch_scores(
            patch_scores,
            method,
        )
        for patch_scores
        in patch_score_vectors
    ]

    return torch.tensor(
        scores,
        dtype=torch.float32,
    )


# ============================================================
# Failure Analysis
# ============================================================

def fn_counts_by_class(
    test_classes,
    test_labels,
    test_scores,
    threshold,
):

    predictions = (
        test_scores >= threshold
    ).long()

    counts = {
        "broken_large": 0,
        "broken_small": 0,
        "contamination": 0,
    }

    for (
        class_name,
        label,
        prediction,
    ) in zip(
        test_classes,
        test_labels.tolist(),
        predictions.tolist(),
    ):

        if (
            label == 1
            and prediction == 0
        ):
            counts[
                class_name
            ] += 1

    return counts


# ============================================================
# Score Ordering Sanity Check
# ============================================================

def check_aggregation_order(
    max_scores,
    top5_scores,
    top20_scores,
    mean_scores,
):

    violations = 0

    for (
        max_score,
        top5_score,
        top20_score,
        mean_score,
    ) in zip(
        max_scores.tolist(),
        top5_scores.tolist(),
        top20_scores.tolist(),
        mean_scores.tolist(),
    ):

        valid = (
            max_score
            >= top5_score
            >= top20_score
            >= mean_score
        )

        if not valid:
            violations += 1

    return violations


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)

    print(
        "EXP020 - Image Score Aggregation Ablation"
    )

    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    # ========================================================
    # 1. Load EXACT EXP018 Split
    # ========================================================

    exp018_path = Path(
        "results/exp018/"
        "threshold_calibration.pt"
    )

    if not exp018_path.exists():
        raise FileNotFoundError(
            f"EXP018 result not found:\n"
            f"{exp018_path.resolve()}"
        )

    exp018 = torch.load(
        exp018_path,
        map_location="cpu",
    )

    reference_paths = [
        Path(path)
        for path
        in exp018["reference_paths"]
    ]

    validation_paths = [
        Path(path)
        for path
        in exp018["validation_paths"]
    ]

    test_paths = [
        Path(path)
        for path
        in exp018["test_paths"]
    ]

    test_labels = (
        exp018["test_labels"]
        .long()
    )

    test_classes = (
        exp018["test_classes"]
    )

    print(
        f"\nReference normal: "
        f"{len(reference_paths)}"
    )

    print(
        f"Validation normal: "
        f"{len(validation_paths)}"
    )

    print(
        f"Test images: "
        f"{len(test_paths)}"
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
    model.to(
        device
    )

    # ========================================================
    # 3. Extract Features ONCE
    # ========================================================

    reference_features = (
        extract_patch_features_for_paths(
            image_paths=
                reference_paths,

            model=
                model,

            preprocess=
                preprocess,

            device=
                device,

            name=
                "reference",
        )
    )

    validation_features = (
        extract_patch_features_for_paths(
            image_paths=
                validation_paths,

            model=
                model,

            preprocess=
                preprocess,

            device=
                device,

            name=
                "validation",
        )
    )

    test_features = (
        extract_patch_features_for_paths(
            image_paths=
                test_paths,

            model=
                model,

            preprocess=
                preprocess,

            device=
                device,

            name=
                "test",
        )
    )

    # ========================================================
    # 4. Build EXACT seed42 Coreset100
    # ========================================================

    full_memory = build_full_memory(
        reference_features
    )

    print(
        "\nFull reference memory:"
    )

    print(
        full_memory.shape
    )

    coreset_result = build_coreset(
        full_memory_bank=
            full_memory,

        seed=42,

        candidate_size=5000,

        coreset_size=100,
    )

    coreset_bank = (
        coreset_result[
            "coreset_bank"
        ]
    )

    print(
        "\nCoreset:"
    )

    print(
        coreset_bank.shape
    )

    print(
        f"Coverage radius: "
        f"{coreset_result['coverage_radius']:.6f}"
    )

    # ========================================================
    # 5. Compute Patch Scores ONCE
    # ========================================================

    print(
        "\nComputing validation patch scores..."
    )

    validation_patch_scores = (
        compute_patch_score_vectors(
            image_features=
                validation_features,

            memory_bank=
                coreset_bank,
        )
    )

    print(
        "Computing test patch scores..."
    )

    test_patch_scores = (
        compute_patch_score_vectors(
            image_features=
                test_features,

            memory_bank=
                coreset_bank,
        )
    )

    # ========================================================
    # 6. Aggregations
    # ========================================================

    methods = [
        "max",
        "top5_mean",
        "top20_mean",
        "mean",
    ]

    validation_scores = {}
    test_scores = {}

    for method in methods:

        validation_scores[
            method
        ] = aggregate_dataset(
            patch_score_vectors=
                validation_patch_scores,

            method=
                method,
        )

        test_scores[
            method
        ] = aggregate_dataset(
            patch_score_vectors=
                test_patch_scores,

            method=
                method,
        )

    # ========================================================
    # 7. Aggregation Ordering Sanity Check
    # ========================================================

    validation_violations = (
        check_aggregation_order(
            validation_scores["max"],
            validation_scores["top5_mean"],
            validation_scores["top20_mean"],
            validation_scores["mean"],
        )
    )

    test_violations = (
        check_aggregation_order(
            test_scores["max"],
            test_scores["top5_mean"],
            test_scores["top20_mean"],
            test_scores["mean"],
        )
    )

    print("\n" + "=" * 70)
    print("Aggregation Ordering Check")
    print("=" * 70)

    print(
        "\nExpected:"
    )

    print(
        "MAX >= Top5 >= Top20 >= Mean"
    )

    print(
        f"\nValidation violations: "
        f"{validation_violations}"
    )

    print(
        f"Test violations: "
        f"{test_violations}"
    )

    # ========================================================
    # 8. Evaluate Each Aggregation
    # ========================================================

    summary_rows = []

    for method in methods:

        print("\n" + "=" * 70)

        print(
            f"Aggregation: {method}"
        )

        print("=" * 70)

        val_scores = (
            validation_scores[
                method
            ]
        )

        current_test_scores = (
            test_scores[
                method
            ]
        )

        # ----------------------------------------------------
        # Calibration
        # ----------------------------------------------------

        p95_threshold = (
            torch.quantile(
                val_scores,
                0.95,
            ).item()
        )

        # ----------------------------------------------------
        # Ranking Metrics
        # ----------------------------------------------------

        overall = binary_metrics(
            labels=
                test_labels,

            scores=
                current_test_scores,
        )

        per_defect = (
            per_defect_metrics(
                test_classes=
                    test_classes,

                test_scores=
                    current_test_scores,
            )
        )

        # ----------------------------------------------------
        # P95 Operating Point
        # ----------------------------------------------------

        confusion = (
            confusion_at_threshold(
                labels=
                    test_labels.tolist(),

                scores=
                    current_test_scores.tolist(),

                threshold=
                    p95_threshold,
            )
        )

        fn_by_class = (
            fn_counts_by_class(
                test_classes=
                    test_classes,

                test_labels=
                    test_labels,

                test_scores=
                    current_test_scores,

                threshold=
                    p95_threshold,
            )
        )

        print(
            f"\nP95 threshold: "
            f"{p95_threshold:.6f}"
        )

        print(
            f"\nOverall AUROC: "
            f"{overall['auroc']:.6f}"
        )

        print(
            f"Overall AP: "
            f"{overall['ap']:.6f}"
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
                f"  {defect_type:<15} "
                f"{per_defect[defect_type]['auroc']:.6f}"
            )

        print(
            "\nP95 Operating Point:"
        )

        print(
            f"  TP = "
            f"{confusion['tp']}"
        )

        print(
            f"  FP = "
            f"{confusion['fp']}"
        )

        print(
            f"  TN = "
            f"{confusion['tn']}"
        )

        print(
            f"  FN = "
            f"{confusion['fn']}"
        )

        print(
            f"  Recall = "
            f"{confusion['recall']:.6f}"
        )

        print(
            f"  FPR = "
            f"{confusion['fpr']:.6f}"
        )

        print(
            f"  F1 = "
            f"{confusion['f1']:.6f}"
        )

        print(
            "\nFN by defect:"
        )

        for defect_type in [
            "broken_large",
            "broken_small",
            "contamination",
        ]:

            print(
                f"  {defect_type:<15} "
                f"{fn_by_class[defect_type]}"
            )

        summary_rows.append(
            {
                "aggregation":
                    method,

                "p95_threshold":
                    p95_threshold,

                "overall_auroc":
                    overall["auroc"],

                "overall_ap":
                    overall["ap"],

                "broken_large_auroc":
                    per_defect[
                        "broken_large"
                    ]["auroc"],

                "broken_small_auroc":
                    per_defect[
                        "broken_small"
                    ]["auroc"],

                "contamination_auroc":
                    per_defect[
                        "contamination"
                    ]["auroc"],

                "p95_tp":
                    confusion["tp"],

                "p95_fp":
                    confusion["fp"],

                "p95_tn":
                    confusion["tn"],

                "p95_fn":
                    confusion["fn"],

                "p95_precision":
                    confusion[
                        "precision"
                    ],

                "p95_recall":
                    confusion[
                        "recall"
                    ],

                "p95_fpr":
                    confusion[
                        "fpr"
                    ],

                "p95_f1":
                    confusion[
                        "f1"
                    ],

                "fn_broken_large":
                    fn_by_class[
                        "broken_large"
                    ],

                "fn_broken_small":
                    fn_by_class[
                        "broken_small"
                    ],

                "fn_contamination":
                    fn_by_class[
                        "contamination"
                    ],
            }
        )

    # ========================================================
    # 9. MAX Reproduction Check
    # ========================================================

    max_row = next(
        row
        for row in summary_rows
        if row["aggregation"] == "max"
    )

    exp018_patch_auroc = float(
        exp018["patch_auroc"]
    )

    delta = abs(
        max_row["overall_auroc"]
        - exp018_patch_auroc
    )

    reproduction_ok = (
        delta < 1e-6
    )

    print("\n" + "=" * 70)

    print(
        "MAX Reproduction Check"
    )

    print("=" * 70)

    print(
        f"\nEXP018 Patch AUROC: "
        f"{exp018_patch_auroc:.6f}"
    )

    print(
        f"EXP020 MAX AUROC: "
        f"{max_row['overall_auroc']:.6f}"
    )

    print(
        f"Absolute delta: "
        f"{delta:.8f}"
    )

    print(
        f"Exact reproduction "
        f"(tol=1e-6): "
        f"{reproduction_ok}"
    )

    if not reproduction_ok:
        raise RuntimeError(
            "MAX failed to reproduce EXP018."
        )

    # ========================================================
    # 10. Save Summary CSV
    # ========================================================

    output_dir = Path(
        "results/exp020"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_csv = (
        output_dir
        / "aggregation_ablation_summary.csv"
    )

    with open(
        summary_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=
                list(
                    summary_rows[0].keys()
                ),
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    # ========================================================
    # 11. Save Per-image Scores
    # ========================================================

    scores_csv = (
        output_dir
        / "aggregation_ablation_scores.csv"
    )

    with open(
        scores_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fieldnames = [
            "path",
            "class",
            "label",
            "max",
            "top5_mean",
            "top20_mean",
            "mean",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for index in range(
            len(test_paths)
        ):

            writer.writerow(
                {
                    "path":
                        str(
                            test_paths[index]
                        ),

                    "class":
                        test_classes[index],

                    "label":
                        int(
                            test_labels[index]
                        ),

                    "max":
                        float(
                            test_scores[
                                "max"
                            ][index]
                        ),

                    "top5_mean":
                        float(
                            test_scores[
                                "top5_mean"
                            ][index]
                        ),

                    "top20_mean":
                        float(
                            test_scores[
                                "top20_mean"
                            ][index]
                        ),

                    "mean":
                        float(
                            test_scores[
                                "mean"
                            ][index]
                        ),
                }
            )

    # ========================================================
    # 12. Save PT Summary
    # ========================================================

    summary_pt = (
        output_dir
        / "aggregation_ablation_summary.pt"
    )

    torch.save(
        {
            "summary_rows":
                summary_rows,

            "validation_scores":
                validation_scores,

            "test_scores":
                test_scores,

            "validation_order_violations":
                validation_violations,

            "test_order_violations":
                test_violations,

            "max_reproduction_delta":
                delta,

            "max_reproduction_ok":
                reproduction_ok,
        },
        summary_pt,
    )

    print("\n" + "=" * 70)
    print("Saved Results")
    print("=" * 70)

    print("\nSummary CSV:")
    print(
        summary_csv.resolve()
    )

    print("\nScores CSV:")
    print(
        scores_csv.resolve()
    )

    print("\nPT:")
    print(
        summary_pt.resolve()
    )

    print("\n" + "=" * 70)
    print("Experiment completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()