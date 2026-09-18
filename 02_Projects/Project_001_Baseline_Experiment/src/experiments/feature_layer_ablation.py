from pathlib import Path
import csv
import statistics

from PIL import Image

import torch
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)

from src.experiments.inspect_patch_features import (
    feature_map_to_patches,
)

from src.experiments.coreset_stability_repeated_seed import (
    build_coreset,
    binary_metrics,
    per_defect_metrics,
)

from src.experiments.threshold_calibration_normal_only import (
    confusion_at_threshold,
)


SEEDS = [42, 43, 44, 45, 46]

LAYERS = [
    "layer2",
    "layer3",
]


# ============================================================
# Multi-layer Feature Extraction
# ============================================================

@torch.inference_mode()
def extract_multilayer_image(
    image_path,
    model,
    preprocess,
    device,
):
    """
    One ResNet forward pass.

    Return:
        layer2 -> [784, 128]
        layer3 -> [196, 256]
    """

    image = Image.open(
        image_path
    ).convert("RGB")

    x = (
        preprocess(image)
        .unsqueeze(0)
        .to(device)
    )

    # Stem
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    # layer1
    x = model.layer1(x)

    # layer2
    layer2_map = model.layer2(x)

    # layer3
    layer3_map = model.layer3(
        layer2_map
    )

    layer2_patches = (
        feature_map_to_patches(
            layer2_map
        )[0]
        .cpu()
    )

    layer3_patches = (
        feature_map_to_patches(
            layer3_map
        )[0]
        .cpu()
    )

    return {
        "layer2":
            layer2_patches,

        "layer3":
            layer3_patches,
    }


def extract_dataset_features(
    image_paths,
    model,
    preprocess,
    device,
    name,
):
    """
    Extract layer2 + layer3 from each image
    in the same forward pass.
    """

    features = {
        "layer2": [],
        "layer3": [],
    }

    print(
        f"\nExtracting {name} features..."
    )

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        result = extract_multilayer_image(
            image_path=image_path,
            model=model,
            preprocess=preprocess,
            device=device,
        )

        for layer in LAYERS:
            features[layer].append(
                result[layer]
            )

        if (
            index % 20 == 0
            or index == len(image_paths)
        ):
            print(
                f"  {name}: "
                f"{index}/{len(image_paths)}"
            )

    return features


# ============================================================
# Memory / Scoring
# ============================================================

def build_full_memory(
    features,
):
    return torch.cat(
        features,
        dim=0,
    )


@torch.inference_mode()
def score_images_max(
    image_features,
    memory_bank,
):
    """
    Image score:
        max patch nearest-neighbor distance.
    """

    scores = []

    for patches in image_features:

        distances = torch.cdist(
            patches,
            memory_bank,
            p=2,
        )

        patch_scores = (
            distances
            .min(dim=1)
            .values
        )

        image_score = (
            patch_scores
            .max()
            .item()
        )

        scores.append(
            image_score
        )

    return torch.tensor(
        scores,
        dtype=torch.float32,
    )


# ============================================================
# Helpers
# ============================================================

def summarize(
    values,
):
    return {
        "mean":
            statistics.mean(values),

        "std":
            statistics.stdev(values),

        "min":
            min(values),

        "max":
            max(values),
    }


def print_summary(
    title,
    values,
):
    s = summarize(values)

    print(
        f"\n{title}:"
    )

    print(
        f"  mean = {s['mean']:.6f}"
    )

    print(
        f"  std  = {s['std']:.6f}"
    )

    print(
        f"  min  = {s['min']:.6f}"
    )

    print(
        f"  max  = {s['max']:.6f}"
    )

    return s


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "EXP021 - Feature Layer Ablation"
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

    print(
        f"Seeds: {SEEDS}"
    )

    # ========================================================
    # 1. Load EXACT EXP018 split
    # ========================================================

    exp018_path = Path(
        "results/exp018/"
        "threshold_calibration.pt"
    )

    if not exp018_path.exists():
        raise FileNotFoundError(
            exp018_path.resolve()
        )

    exp018 = torch.load(
        exp018_path,
        map_location="cpu",
    )

    reference_paths = [
        Path(p)
        for p in exp018[
            "reference_paths"
        ]
    ]

    validation_paths = [
        Path(p)
        for p in exp018[
            "validation_paths"
        ]
    ]

    test_paths = [
        Path(p)
        for p in exp018[
            "test_paths"
        ]
    ]

    test_labels = (
        exp018[
            "test_labels"
        ].long()
    )

    test_classes = (
        exp018[
            "test_classes"
        ]
    )

    print(
        f"\nReference: "
        f"{len(reference_paths)}"
    )

    print(
        f"Validation: "
        f"{len(validation_paths)}"
    )

    print(
        f"Test: "
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
    model.to(device)

    # ========================================================
    # 3. Extract both layers ONCE
    # ========================================================

    reference_features = (
        extract_dataset_features(
            reference_paths,
            model,
            preprocess,
            device,
            "reference",
        )
    )

    validation_features = (
        extract_dataset_features(
            validation_paths,
            model,
            preprocess,
            device,
            "validation",
        )
    )

    test_features = (
        extract_dataset_features(
            test_paths,
            model,
            preprocess,
            device,
            "test",
        )
    )

    # ========================================================
    # 4. Shape sanity check
    # ========================================================

    print("\n" + "=" * 70)
    print("Feature Shape Check")
    print("=" * 70)

    for layer in LAYERS:

        shape = (
            reference_features[
                layer
            ][0].shape
        )

        print(
            f"{layer}: {shape}"
        )

    expected_shapes = {
        "layer2":
            (784, 128),

        "layer3":
            (196, 256),
    }

    for layer in LAYERS:

        actual = tuple(
            reference_features[
                layer
            ][0].shape
        )

        if (
            actual
            != expected_shapes[layer]
        ):
            raise RuntimeError(
                f"{layer} unexpected "
                f"shape: {actual}"
            )

    # ========================================================
    # 5. Evaluate layer × seed
    # ========================================================

    rows = []

    for layer in LAYERS:

        print("\n" + "=" * 70)

        print(
            f"Feature Layer: {layer}"
        )

        print("=" * 70)

        full_memory = (
            build_full_memory(
                reference_features[
                    layer
                ]
            )
        )

        num_patches = (
            reference_features[
                layer
            ][0].shape[0]
        )

        feature_dim = (
            reference_features[
                layer
            ][0].shape[1]
        )

        expected_rows = (
            len(reference_paths)
            * num_patches
        )

        print(
            f"\nPatch count/image: "
            f"{num_patches}"
        )

        print(
            f"Feature dim: "
            f"{feature_dim}"
        )

        print(
            f"Full memory: "
            f"{tuple(full_memory.shape)}"
        )

        if (
            full_memory.shape[0]
            != expected_rows
        ):
            raise RuntimeError(
                "Unexpected memory size."
            )

        for seed in SEEDS:

            print("\n" + "-" * 70)

            print(
                f"{layer} | seed={seed}"
            )

            print("-" * 70)

            result = build_coreset(
                full_memory_bank=
                    full_memory,

                seed=seed,

                candidate_size=5000,

                coreset_size=100,
            )

            coreset = result[
                "coreset_bank"
            ]

            # -----------------------------------------------
            # Validation calibration
            # -----------------------------------------------

            val_scores = score_images_max(
                validation_features[
                    layer
                ],
                coreset,
            )

            threshold = (
                torch.quantile(
                    val_scores,
                    0.95,
                ).item()
            )

            # -----------------------------------------------
            # Test
            # -----------------------------------------------

            test_scores = score_images_max(
                test_features[
                    layer
                ],
                coreset,
            )

            overall = binary_metrics(
                labels=
                    test_labels,

                scores=
                    test_scores,
            )

            per_defect = (
                per_defect_metrics(
                    test_classes=
                        test_classes,

                    test_scores=
                        test_scores,
                )
            )

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

            print(
                f"\nAUROC: "
                f"{overall['auroc']:.6f}"
            )

            print(
                f"AP: "
                f"{overall['ap']:.6f}"
            )

            print(
                f"P95 threshold: "
                f"{threshold:.6f}"
            )

            print(
                "\nPer-defect AUROC:"
            )

            for defect in [
                "broken_large",
                "broken_small",
                "contamination",
            ]:
                print(
                    f"  {defect:<15} "
                    f"{per_defect[defect]['auroc']:.6f}"
                )

            print(
                "\nP95:"
            )

            print(
                f"  TP={confusion['tp']} "
                f"FP={confusion['fp']} "
                f"TN={confusion['tn']} "
                f"FN={confusion['fn']}"
            )

            print(
                f"  Recall="
                f"{confusion['recall']:.6f}"
            )

            print(
                f"  FPR="
                f"{confusion['fpr']:.6f}"
            )

            rows.append(
                {
                    "layer":
                        layer,

                    "seed":
                        seed,

                    "patches_per_image":
                        num_patches,

                    "feature_dim":
                        feature_dim,

                    "memory_rows":
                        full_memory.shape[0],

                    "coverage_radius":
                        result[
                            "coverage_radius"
                        ],

                    "p95_threshold":
                        threshold,

                    "overall_auroc":
                        overall[
                            "auroc"
                        ],

                    "overall_ap":
                        overall[
                            "ap"
                        ],

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
                }
            )

    # ========================================================
    # 6. Layer2 reproduction check
    # ========================================================

    l2_seed42 = next(
        row
        for row in rows
        if (
            row["layer"] == "layer2"
            and row["seed"] == 42
        )
    )

    exp018_auroc = float(
        exp018["patch_auroc"]
    )

    reproduction_delta = abs(
        l2_seed42[
            "overall_auroc"
        ]
        - exp018_auroc
    )

    reproduction_ok = (
        reproduction_delta
        < 1e-6
    )

    print("\n" + "=" * 70)
    print("Layer2 Seed42 Reproduction")
    print("=" * 70)

    print(
        f"\nEXP018: "
        f"{exp018_auroc:.6f}"
    )

    print(
        f"EXP021 layer2 seed42: "
        f"{l2_seed42['overall_auroc']:.6f}"
    )

    print(
        f"Delta: "
        f"{reproduction_delta:.8f}"
    )

    print(
        f"PASS: "
        f"{reproduction_ok}"
    )

    if not reproduction_ok:
        raise RuntimeError(
            "Layer2 reproduction failed."
        )

    # ========================================================
    # 7. Repeated-seed summaries
    # ========================================================

    summary_rows = []

    print("\n" + "=" * 70)
    print("Repeated Seed Summary")
    print("=" * 70)

    metrics = [
        "overall_auroc",
        "overall_ap",
        "broken_large_auroc",
        "broken_small_auroc",
        "contamination_auroc",
        "p95_recall",
        "p95_fpr",
    ]

    for layer in LAYERS:

        layer_rows = [
            row
            for row in rows
            if row["layer"] == layer
        ]

        print(
            f"\n### {layer}"
        )

        summary = {
            "layer":
                layer,
        }

        for metric in metrics:

            values = [
                row[metric]
                for row
                in layer_rows
            ]

            stats = print_summary(
                metric,
                values,
            )

            for key in [
                "mean",
                "std",
                "min",
                "max",
            ]:
                summary[
                    f"{metric}_{key}"
                ] = stats[key]

        summary_rows.append(
            summary
        )

    # ========================================================
    # 8. Mean AUROC difference
    # ========================================================

    l2_summary = next(
        row
        for row in summary_rows
        if row["layer"] == "layer2"
    )

    l3_summary = next(
        row
        for row in summary_rows
        if row["layer"] == "layer3"
    )

    mean_delta = (
        l3_summary[
            "overall_auroc_mean"
        ]
        - l2_summary[
            "overall_auroc_mean"
        ]
    )

    print("\n" + "=" * 70)
    print("Layer Mean Difference")
    print("=" * 70)

    print(
        "\nLayer3 mean AUROC "
        "- Layer2 mean AUROC:"
    )

    print(
        f"{mean_delta:+.6f}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Do NOT compare raw thresholds "
        "or coverage radii across layers "
        "as if they share the same scale."
    )

    # ========================================================
    # 9. Save
    # ========================================================

    output_dir = Path(
        "results/exp021"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs_csv = (
        output_dir
        / "feature_layer_runs.csv"
    )

    with open(
        runs_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=
                list(
                    rows[0].keys()
                ),
        )

        writer.writeheader()
        writer.writerows(rows)

    summary_csv = (
        output_dir
        / "feature_layer_summary.csv"
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
                    summary_rows[0]
                    .keys()
                ),
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    torch.save(
        {
            "runs":
                rows,

            "summary":
                summary_rows,

            "layer2_seed42_reproduction":
                reproduction_ok,

            "layer2_seed42_delta":
                reproduction_delta,

            "layer3_minus_layer2_mean_auroc":
                mean_delta,
        },
        output_dir
        / "feature_layer_ablation.pt",
    )

    print("\n" + "=" * 70)
    print("Saved")
    print("=" * 70)

    print(
        runs_csv.resolve()
    )

    print(
        summary_csv.resolve()
    )

    print(
        "\nExperiment completed."
    )


if __name__ == "__main__":
    main()