from pathlib import Path
import csv
import statistics

from PIL import Image

import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)

from src.experiments.full_image_patch_scoring import (
    extract_image_patches,
)

from src.experiments.coreset_minimal_experiment import (
    kcenter_greedy,
)

from src.experiments.threshold_calibration_normal_only import (
    confusion_at_threshold,
)


# ============================================================
# Feature Extraction
# ============================================================

@torch.inference_mode()
def extract_patch_features_for_paths(
    image_paths,
    model,
    preprocess,
    device,
    name,
):
    """
    Extract [784,128] layer2 patch features
    for every image exactly once.
    """

    all_features = []

    print(
        f"\nExtracting {name} patch features..."
    )

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        patches, _, _ = extract_image_patches(
            image_path=Path(image_path),
            model=model,
            preprocess=preprocess,
            device=device,
        )

        all_features.append(
            patches.cpu()
        )

        if (
            index % 20 == 0
            or index == len(image_paths)
        ):
            print(
                f"  {name}: "
                f"{index}/{len(image_paths)}"
            )

    return all_features


def build_full_memory(
    reference_features,
):
    """
    reference_features:
        list of [784,128]

    output:
        [N_reference * 784,128]
    """

    return torch.cat(
        reference_features,
        dim=0,
    )


# ============================================================
# Coreset
# ============================================================

def build_coreset(
    full_memory_bank,
    seed,
    candidate_size=5000,
    coreset_size=100,
):

    generator = torch.Generator()
    generator.manual_seed(
        seed
    )

    # --------------------------------------------------------
    # Candidate sampling
    # --------------------------------------------------------

    permutation = torch.randperm(
        full_memory_bank.shape[0],
        generator=generator,
    )

    candidate_indices = permutation[
        :candidate_size
    ]

    candidate_bank = full_memory_bank[
        candidate_indices
    ]

    # --------------------------------------------------------
    # Initial center
    #
    # Match EXP018 logic:
    # another randperm using the SAME generator state.
    # --------------------------------------------------------

    random_order = torch.randperm(
        candidate_size,
        generator=generator,
    )

    first_index = (
        random_order[0].item()
    )

    # --------------------------------------------------------
    # K-center
    # --------------------------------------------------------

    coreset_indices = kcenter_greedy(
        candidates=candidate_bank,
        num_selected=coreset_size,
        first_index=first_index,
    )

    coreset_bank = candidate_bank[
        coreset_indices
    ]

    # --------------------------------------------------------
    # Candidate coverage radius
    #
    # R(C) = max_x min_c ||x-c||
    # --------------------------------------------------------

    distances = torch.cdist(
        candidate_bank,
        coreset_bank,
        p=2,
    )

    min_distances = distances.min(
        dim=1
    ).values

    coverage_radius = (
        min_distances.max().item()
    )

    return {
        "candidate_bank":
            candidate_bank,

        "coreset_bank":
            coreset_bank,

        "candidate_indices":
            candidate_indices,

        "coreset_indices":
            coreset_indices,

        "first_index":
            first_index,

        "coverage_radius":
            coverage_radius,
    }


# ============================================================
# Image Scoring
# ============================================================

@torch.inference_mode()
def score_images(
    image_features,
    memory_bank,
):
    """
    image_features:
        list of [784,128]

    return:
        [N_images]
    """

    image_scores = []

    for patches in image_features:

        distances = torch.cdist(
            patches,
            memory_bank,
            p=2,
        )

        patch_scores = distances.min(
            dim=1
        ).values

        image_score = patch_scores.max()

        image_scores.append(
            image_score.item()
        )

    return torch.tensor(
        image_scores,
        dtype=torch.float32,
    )


# ============================================================
# Metrics
# ============================================================

def binary_metrics(
    labels,
    scores,
):

    labels_np = labels.numpy()
    scores_np = scores.numpy()

    return {
        "auroc":
            float(
                roc_auc_score(
                    labels_np,
                    scores_np,
                )
            ),

        "ap":
            float(
                average_precision_score(
                    labels_np,
                    scores_np,
                )
            ),
    }


def per_defect_metrics(
    test_classes,
    test_scores,
):

    results = {}

    defect_types = [
        "broken_large",
        "broken_small",
        "contamination",
    ]

    for defect_type in defect_types:

        indices = [
            index
            for index, class_name
            in enumerate(test_classes)
            if (
                class_name == "good"
                or
                class_name == defect_type
            )
        ]

        labels = torch.tensor(
            [
                0
                if test_classes[index] == "good"
                else 1
                for index in indices
            ],
            dtype=torch.long,
        )

        scores = test_scores[
            indices
        ]

        results[defect_type] = (
            binary_metrics(
                labels,
                scores,
            )
        )

    return results


# ============================================================
# Summary Helpers
# ============================================================

def summarize_values(
    values,
):

    return {
        "mean":
            statistics.mean(values),

        "std":
            statistics.stdev(values)
            if len(values) > 1
            else 0.0,

        "min":
            min(values),

        "max":
            max(values),
    }


def print_summary(
    name,
    values,
):

    summary = summarize_values(
        values
    )

    print(
        f"\n{name}:"
    )

    print(
        f"  mean = "
        f"{summary['mean']:.6f}"
    )

    print(
        f"  std  = "
        f"{summary['std']:.6f}"
    )

    print(
        f"  min  = "
        f"{summary['min']:.6f}"
    )

    print(
        f"  max  = "
        f"{summary['max']:.6f}"
    )

    return summary


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)

    print(
        "EXP019 - Coreset Stability / Repeated Seed Ablation"
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

    seeds = [
        42,
        43,
        44,
        45,
        46,
    ]

    print(
        f"Seeds: {seeds}"
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

    test_classes = exp018[
        "test_classes"
    ]

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
    #
    # This is critical:
    # repeated seeds should NOT repeat backbone inference.
    # ========================================================

    reference_features = (
        extract_patch_features_for_paths(
            image_paths=
                reference_paths,

            model=model,

            preprocess=
                preprocess,

            device=device,

            name=
                "reference",
        )
    )

    validation_features = (
        extract_patch_features_for_paths(
            image_paths=
                validation_paths,

            model=model,

            preprocess=
                preprocess,

            device=device,

            name=
                "validation",
        )
    )

    test_features = (
        extract_patch_features_for_paths(
            image_paths=
                test_paths,

            model=model,

            preprocess=
                preprocess,

            device=device,

            name=
                "test",
        )
    )

    # ========================================================
    # 4. Full Reference Memory
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

    expected_rows = (
        len(reference_paths)
        * 784
    )

    print(
        f"Expected rows: "
        f"{expected_rows}"
    )

    if (
        full_memory.shape[0]
        != expected_rows
    ):
        raise RuntimeError(
            "Unexpected patch memory size."
        )

    # ========================================================
    # 5. Repeat Seeds
    # ========================================================

    run_rows = []

    for run_index, seed in enumerate(
        seeds,
        start=1,
    ):

        print("\n" + "=" * 70)

        print(
            f"Run {run_index}/{len(seeds)} "
            f"| seed={seed}"
        )

        print("=" * 70)

        coreset_result = build_coreset(
            full_memory_bank=
                full_memory,

            seed=seed,

            candidate_size=5000,

            coreset_size=100,
        )

        coreset_bank = coreset_result[
            "coreset_bank"
        ]

        coverage_radius = (
            coreset_result[
                "coverage_radius"
            ]
        )

        print(
            f"\nCoverage radius: "
            f"{coverage_radius:.6f}"
        )

        # ----------------------------------------------------
        # Validation Normal Scores
        # ----------------------------------------------------

        val_scores = score_images(
            image_features=
                validation_features,

            memory_bank=
                coreset_bank,
        )

        p95_threshold = (
            torch.quantile(
                val_scores,
                0.95,
            ).item()
        )

        print(
            f"P95 threshold: "
            f"{p95_threshold:.6f}"
        )

        # ----------------------------------------------------
        # Test Scores
        # ----------------------------------------------------

        test_scores = score_images(
            image_features=
                test_features,

            memory_bank=
                coreset_bank,
        )

        # ----------------------------------------------------
        # Overall Ranking Metrics
        # ----------------------------------------------------

        overall = binary_metrics(
            labels=
                test_labels,

            scores=
                test_scores,
        )

        per_defect = per_defect_metrics(
            test_classes=
                test_classes,

            test_scores=
                test_scores,
        )

        # ----------------------------------------------------
        # P95 Operating Point
        # ----------------------------------------------------

        p95_confusion = (
            confusion_at_threshold(
                labels=
                    test_labels.tolist(),

                scores=
                    test_scores.tolist(),

                threshold=
                    p95_threshold,
            )
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
            f"{p95_confusion['tp']}"
        )

        print(
            f"  FP = "
            f"{p95_confusion['fp']}"
        )

        print(
            f"  FN = "
            f"{p95_confusion['fn']}"
        )

        print(
            f"  Recall = "
            f"{p95_confusion['recall']:.6f}"
        )

        print(
            f"  FPR = "
            f"{p95_confusion['fpr']:.6f}"
        )

        print(
            f"  F1 = "
            f"{p95_confusion['f1']:.6f}"
        )

        run_rows.append(
            {
                "seed":
                    seed,

                "coverage_radius":
                    coverage_radius,

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

                "broken_large_ap":
                    per_defect[
                        "broken_large"
                    ]["ap"],

                "broken_small_auroc":
                    per_defect[
                        "broken_small"
                    ]["auroc"],

                "broken_small_ap":
                    per_defect[
                        "broken_small"
                    ]["ap"],

                "contamination_auroc":
                    per_defect[
                        "contamination"
                    ]["auroc"],

                "contamination_ap":
                    per_defect[
                        "contamination"
                    ]["ap"],

                "p95_tp":
                    p95_confusion["tp"],

                "p95_fp":
                    p95_confusion["fp"],

                "p95_tn":
                    p95_confusion["tn"],

                "p95_fn":
                    p95_confusion["fn"],

                "p95_precision":
                    p95_confusion[
                        "precision"
                    ],

                "p95_recall":
                    p95_confusion[
                        "recall"
                    ],

                "p95_fpr":
                    p95_confusion[
                        "fpr"
                    ],

                "p95_f1":
                    p95_confusion[
                        "f1"
                    ],
            }
        )

    # ========================================================
    # 6. Seed-42 Reproduction Check
    # ========================================================

    seed42 = next(
        row
        for row in run_rows
        if row["seed"] == 42
    )

    exp018_auroc = float(
        exp018["patch_auroc"]
    )

    reproduction_delta = abs(
        seed42["overall_auroc"]
        - exp018_auroc
    )

    print("\n" + "=" * 70)

    print(
        "Seed-42 Reproduction Check"
    )

    print("=" * 70)

    print(
        f"\nEXP018 AUROC: "
        f"{exp018_auroc:.6f}"
    )

    print(
        f"EXP019 seed42: "
        f"{seed42['overall_auroc']:.6f}"
    )

    print(
        f"Absolute delta: "
        f"{reproduction_delta:.8f}"
    )

    reproduction_ok = (
        reproduction_delta
        < 1e-6
    )

    print(
        f"Exact reproduction "
        f"(tol=1e-6): "
        f"{reproduction_ok}"
    )

    # ========================================================
    # 7. Stability Summary
    # ========================================================

    print("\n" + "=" * 70)
    print("Stability Summary")
    print("=" * 70)

    overall_auroc_summary = (
        print_summary(
            "Overall AUROC",
            [
                row["overall_auroc"]
                for row in run_rows
            ],
        )
    )

    overall_ap_summary = (
        print_summary(
            "Overall AP",
            [
                row["overall_ap"]
                for row in run_rows
            ],
        )
    )

    coverage_summary = (
        print_summary(
            "Coverage Radius",
            [
                row["coverage_radius"]
                for row in run_rows
            ],
        )
    )

    p95_threshold_summary = (
        print_summary(
            "P95 Threshold",
            [
                row["p95_threshold"]
                for row in run_rows
            ],
        )
    )

    p95_recall_summary = (
        print_summary(
            "P95 Recall",
            [
                row["p95_recall"]
                for row in run_rows
            ],
        )
    )

    p95_fpr_summary = (
        print_summary(
            "P95 FPR",
            [
                row["p95_fpr"]
                for row in run_rows
            ],
        )
    )

    defect_summaries = {}

    for defect_type in [
        "broken_large",
        "broken_small",
        "contamination",
    ]:

        defect_summaries[
            defect_type
        ] = print_summary(
            f"{defect_type} AUROC",
            [
                row[
                    f"{defect_type}_auroc"
                ]
                for row in run_rows
            ],
        )

    # ========================================================
    # 8. Save CSV
    # ========================================================

    output_dir = Path(
        "results/exp019"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "coreset_stability_runs.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=
                list(
                    run_rows[0].keys()
                ),
        )

        writer.writeheader()

        writer.writerows(
            run_rows
        )

    # ========================================================
    # 9. Save Summary
    # ========================================================

    summary_path = (
        output_dir
        / "coreset_stability_summary.pt"
    )

    torch.save(
        {
            "seeds":
                seeds,

            "runs":
                run_rows,

            "seed42_reproduction_delta":
                reproduction_delta,

            "seed42_reproduction_ok":
                reproduction_ok,

            "overall_auroc":
                overall_auroc_summary,

            "overall_ap":
                overall_ap_summary,

            "coverage_radius":
                coverage_summary,

            "p95_threshold":
                p95_threshold_summary,

            "p95_recall":
                p95_recall_summary,

            "p95_fpr":
                p95_fpr_summary,

            "per_defect":
                defect_summaries,
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