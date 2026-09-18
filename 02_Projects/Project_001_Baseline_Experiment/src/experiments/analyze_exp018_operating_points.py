from pathlib import Path
import csv
from collections import Counter

import torch

from src.experiments.threshold_calibration_normal_only import (
    confusion_at_threshold,
)


def evaluate_failures(
    test_paths,
    test_classes,
    labels,
    scores,
    threshold,
):
    """
    Return exact FP / FN samples and
    FN counts grouped by defect type.
    """

    predictions = (
        scores >= threshold
    ).long()

    false_positives = []
    false_negatives = []

    for (
        path,
        class_name,
        label,
        prediction,
        score,
    ) in zip(
        test_paths,
        test_classes,
        labels.tolist(),
        predictions.tolist(),
        scores.tolist(),
    ):

        record = {
            "path": path,
            "class": class_name,
            "label": label,
            "prediction": prediction,
            "score": score,
        }

        # Good predicted as anomaly
        if (
            label == 0
            and prediction == 1
        ):
            false_positives.append(
                record
            )

        # Defect predicted as good
        if (
            label == 1
            and prediction == 0
        ):
            false_negatives.append(
                record
            )

    fn_by_class = Counter(
        item["class"]
        for item in false_negatives
    )

    return {
        "false_positives":
            false_positives,

        "false_negatives":
            false_negatives,

        "fn_by_class":
            dict(fn_by_class),
    }


def print_operating_point(
    method_name,
    percentile_name,
    threshold,
    confusion,
    failures,
):
    print("\n" + "=" * 70)

    print(
        f"{method_name} - {percentile_name}"
    )

    print("=" * 70)

    print(
        f"\nThreshold = "
        f"{threshold:.6f}"
    )

    print(
        f"\nTP = {confusion['tp']}"
    )

    print(
        f"FP = {confusion['fp']}"
    )

    print(
        f"TN = {confusion['tn']}"
    )

    print(
        f"FN = {confusion['fn']}"
    )

    print(
        f"\nPrecision = "
        f"{confusion['precision']:.6f}"
    )

    print(
        f"Recall    = "
        f"{confusion['recall']:.6f}"
    )

    print(
        f"FPR       = "
        f"{confusion['fpr']:.6f}"
    )

    print(
        f"F1        = "
        f"{confusion['f1']:.6f}"
    )

    print(
        "\nFN by defect type:"
    )

    for defect_type in [
        "broken_large",
        "broken_small",
        "contamination",
    ]:

        count = failures[
            "fn_by_class"
        ].get(
            defect_type,
            0,
        )

        print(
            f"  {defect_type:<15} "
            f"{count}"
        )

    print(
        "\nFalse Positive samples:"
    )

    if not failures[
        "false_positives"
    ]:
        print("  None")

    else:
        for item in failures[
            "false_positives"
        ]:

            print(
                f"  {item['path']} "
                f"score="
                f"{item['score']:.6f}"
            )

    print(
        "\nFalse Negative samples:"
    )

    if not failures[
        "false_negatives"
    ]:
        print("  None")

    else:
        for item in failures[
            "false_negatives"
        ]:

            print(
                f"  {item['class']}/"
                f"{Path(item['path']).name} "
                f"score="
                f"{item['score']:.6f}"
            )


def analyze_method(
    method_name,
    validation_scores,
    test_scores,
    test_labels,
    test_paths,
    test_classes,
):

    thresholds = {
        "P90":
            torch.quantile(
                validation_scores,
                0.90,
            ).item(),

        "P95":
            torch.quantile(
                validation_scores,
                0.95,
            ).item(),

        "P99":
            torch.quantile(
                validation_scores,
                0.99,
            ).item(),
    }

    print("\n" + "#" * 70)
    print(method_name)
    print("#" * 70)

    print("\nThresholds:")

    for name, threshold in (
        thresholds.items()
    ):
        print(
            f"  {name}: "
            f"{threshold:.6f}"
        )

    print(
        "\nThreshold ordering:"
    )

    ordering_ok = (
        thresholds["P90"]
        <= thresholds["P95"]
        <= thresholds["P99"]
    )

    print(
        f"  P90 <= P95 <= P99: "
        f"{ordering_ok}"
    )

    results = {}

    for percentile_name, threshold in (
        thresholds.items()
    ):

        confusion = (
            confusion_at_threshold(
                labels=
                    test_labels.tolist(),

                scores=
                    test_scores.tolist(),

                threshold=
                    threshold,
            )
        )

        failures = evaluate_failures(
            test_paths=
                test_paths,

            test_classes=
                test_classes,

            labels=
                test_labels,

            scores=
                test_scores,

            threshold=
                threshold,
        )

        print_operating_point(
            method_name=
                method_name,

            percentile_name=
                percentile_name,

            threshold=
                threshold,

            confusion=
                confusion,

            failures=
                failures,
        )

        results[
            percentile_name
        ] = {
            "threshold":
                threshold,

            "confusion":
                confusion,

            "failures":
                failures,
        }

    # --------------------------------------------------------
    # Monotonicity checks
    # --------------------------------------------------------

    p90 = results[
        "P90"
    ]["confusion"]

    p95 = results[
        "P95"
    ]["confusion"]

    p99 = results[
        "P99"
    ]["confusion"]

    print("\n" + "-" * 70)

    print(
        f"{method_name} "
        f"Monotonicity"
    )

    print("-" * 70)

    print(
        "TP90 >= TP95 >= TP99: "
        f"{p90['tp'] >= p95['tp'] >= p99['tp']}"
    )

    print(
        "FP90 >= FP95 >= FP99: "
        f"{p90['fp'] >= p95['fp'] >= p99['fp']}"
    )

    print(
        "FN90 <= FN95 <= FN99: "
        f"{p90['fn'] <= p95['fn'] <= p99['fn']}"
    )

    print(
        "Recall90 >= Recall95 >= Recall99: "
        f"{p90['recall'] >= p95['recall'] >= p99['recall']}"
    )

    print(
        "FPR90 >= FPR95 >= FPR99: "
        f"{p90['fpr'] >= p95['fpr'] >= p99['fpr']}"
    )

    return results


def main():

    print("=" * 70)

    print(
        "EXP018 - Operating Point Analysis"
    )

    print("=" * 70)

    # ========================================================
    # 1. Load existing EXP018 result
    # ========================================================

    result_path = Path(
        "results/exp018/"
        "threshold_calibration.pt"
    )

    if not result_path.exists():
        raise FileNotFoundError(
            f"EXP018 result not found:\n"
            f"{result_path.resolve()}"
        )

    result = torch.load(
        result_path,
        map_location="cpu",
    )

    # ========================================================
    # 2. Restore saved data
    # ========================================================

    test_paths = result[
        "test_paths"
    ]

    test_classes = result[
        "test_classes"
    ]

    test_labels = result[
        "test_labels"
    ].long()

    global_val_scores = result[
        "global_val_scores"
    ].float()

    patch_val_scores = result[
        "patch_val_scores"
    ].float()

    global_test_scores = result[
        "global_test_scores"
    ].float()

    patch_test_scores = result[
        "patch_test_scores"
    ].float()

    print(
        f"\nTest samples: "
        f"{len(test_paths)}"
    )

    print(
        f"Global validation samples: "
        f"{len(global_val_scores)}"
    )

    print(
        f"Patch validation samples: "
        f"{len(patch_val_scores)}"
    )

    # ========================================================
    # 3. Global Operating Points
    # ========================================================

    global_results = analyze_method(
        method_name=
            "GLOBAL",

        validation_scores=
            global_val_scores,

        test_scores=
            global_test_scores,

        test_labels=
            test_labels,

        test_paths=
            test_paths,

        test_classes=
            test_classes,
    )

    # ========================================================
    # 4. Patch Operating Points
    # ========================================================

    patch_results = analyze_method(
        method_name=
            "PATCH",

        validation_scores=
            patch_val_scores,

        test_scores=
            patch_test_scores,

        test_labels=
            test_labels,

        test_paths=
            test_paths,

        test_classes=
            test_classes,
    )

    # ========================================================
    # 5. Save per-image decisions as CSV
    # ========================================================

    output_dir = Path(
        "results/exp018"
    )

    csv_path = (
        output_dir
        / "operating_point_predictions.csv"
    )

    fieldnames = [
        "path",
        "class",
        "label",

        "global_score",
        "global_p90",
        "global_p95",
        "global_p99",

        "patch_score",
        "patch_p90",
        "patch_p95",
        "patch_p99",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for index in range(
            len(test_paths)
        ):

            global_score = float(
                global_test_scores[
                    index
                ]
            )

            patch_score = float(
                patch_test_scores[
                    index
                ]
            )

            writer.writerow(
                {
                    "path":
                        test_paths[index],

                    "class":
                        test_classes[index],

                    "label":
                        int(
                            test_labels[index]
                        ),

                    "global_score":
                        global_score,

                    "global_p90":
                        int(
                            global_score
                            >= global_results[
                                "P90"
                            ]["threshold"]
                        ),

                    "global_p95":
                        int(
                            global_score
                            >= global_results[
                                "P95"
                            ]["threshold"]
                        ),

                    "global_p99":
                        int(
                            global_score
                            >= global_results[
                                "P99"
                            ]["threshold"]
                        ),

                    "patch_score":
                        patch_score,

                    "patch_p90":
                        int(
                            patch_score
                            >= patch_results[
                                "P90"
                            ]["threshold"]
                        ),

                    "patch_p95":
                        int(
                            patch_score
                            >= patch_results[
                                "P95"
                            ]["threshold"]
                        ),

                    "patch_p99":
                        int(
                            patch_score
                            >= patch_results[
                                "P99"
                            ]["threshold"]
                        ),
                }
            )

    # ========================================================
    # 6. Save summary
    # ========================================================

    output_path = (
        output_dir
        / "operating_point_analysis.pt"
    )

    torch.save(
        {
            "global":
                global_results,

            "patch":
                patch_results,
        },
        output_path,
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
        output_path.resolve()
    )

    print("\n" + "=" * 70)
    print("Experiment completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()