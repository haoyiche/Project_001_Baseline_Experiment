from pathlib import Path
import csv
import statistics
import time

from PIL import Image

import torch
from torchvision.models import ResNet18_Weights

from src.experiments.engineering_benchmark import (
    build_global_model,
    build_patch_model,
    build_global_center,
    build_patch_reference,
    preprocess_paths,
    global_score,
    patch_score,
    synchronize,
    summarize_latencies,
    calculate_accuracy,
)


# ============================================================
# Configuration
# ============================================================

BENCHMARK_REPEATS = 5
WARMUP_PAIRS = 20


# ============================================================
# One Complete End-to-End Request
# ============================================================

def measure_e2e_once(
    path,
    preprocess,
    score_function,
    device,
):
    """
    Measurement boundary:

    disk PNG
    -> PIL decode / RGB
    -> preprocess
    -> CPU tensor
    -> GPU
    -> backbone
    -> anomaly score
    """

    # Ensure previous GPU work does not leak
    # into the current measurement.
    synchronize(device)

    start = time.perf_counter()

    with Image.open(path) as image:
        image = image.convert("RGB")
        tensor = preprocess(image)

    score_tensor = score_function(
        tensor
    )

    # Wait until GPU computation is really done.
    synchronize(device)

    elapsed_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    # Keep .item() outside measured region,
    # consistent with previous EXP022 benchmark.
    score = score_tensor.item()

    return elapsed_ms, score


# ============================================================
# Warm-up
# ============================================================

def warmup_interleaved(
    test_paths,
    preprocess,
    global_score_function,
    patch_score_function,
    device,
):
    print(
        f"\nWarm-up: {WARMUP_PAIRS} balanced pairs"
    )

    for index in range(WARMUP_PAIRS):

        path = test_paths[
            index % len(test_paths)
        ]

        # Alternate order during warm-up too.
        if index % 2 == 0:
            methods = [
                (
                    "global",
                    global_score_function,
                ),
                (
                    "patch_layer3",
                    patch_score_function,
                ),
            ]
        else:
            methods = [
                (
                    "patch_layer3",
                    patch_score_function,
                ),
                (
                    "global",
                    global_score_function,
                ),
            ]

        for _, score_function in methods:

            measure_e2e_once(
                path=path,
                preprocess=preprocess,
                score_function=score_function,
                device=device,
            )

    synchronize(device)


# ============================================================
# Interleaved Benchmark
# ============================================================

def benchmark_interleaved(
    test_paths,
    preprocess,
    global_score_function,
    patch_score_function,
    device,
):
    rows = []

    # Scores from repeat 0 are enough
    # for accuracy regression checks.
    first_repeat_scores = {
        "global": [],
        "patch_layer3": [],
    }

    for repeat in range(
        BENCHMARK_REPEATS
    ):

        print(
            f"\nRepeat "
            f"{repeat + 1}/"
            f"{BENCHMARK_REPEATS}"
        )

        for image_index, path in enumerate(
            test_paths
        ):

            # Alternate which method runs first.
            #
            # Example:
            #
            # repeat 0, image 0:
            # Global -> Patch
            #
            # repeat 0, image 1:
            # Patch -> Global
            #
            # repeat 1, image 0:
            # Patch -> Global
            #
            # This balances both image position
            # and repeat position.
            global_first = (
                (repeat + image_index)
                % 2
                == 0
            )

            if global_first:

                methods = [
                    (
                        "global",
                        global_score_function,
                    ),
                    (
                        "patch_layer3",
                        patch_score_function,
                    ),
                ]

            else:

                methods = [
                    (
                        "patch_layer3",
                        patch_score_function,
                    ),
                    (
                        "global",
                        global_score_function,
                    ),
                ]

            order_name = (
                "global_then_patch"
                if global_first
                else "patch_then_global"
            )

            for order_position, (
                method_name,
                score_function,
            ) in enumerate(
                methods,
                start=1,
            ):

                latency_ms, score = (
                    measure_e2e_once(
                        path=path,
                        preprocess=preprocess,
                        score_function=score_function,
                        device=device,
                    )
                )

                rows.append(
                    {
                        "repeat":
                            repeat,

                        "image_index":
                            image_index,

                        "path":
                            str(path),

                        "pair_order":
                            order_name,

                        "method":
                            method_name,

                        "order_position":
                            order_position,

                        "latency_ms":
                            latency_ms,

                        "score":
                            score,
                    }
                )

                if repeat == 0:

                    first_repeat_scores[
                        method_name
                    ].append(
                        score
                    )

    return (
        rows,
        first_repeat_scores,
    )


# ============================================================
# Analysis
# ============================================================

def analyze_method(
    rows,
    method_name,
):
    method_rows = [
        row
        for row in rows
        if row["method"] == method_name
    ]

    latencies = [
        row["latency_ms"]
        for row in method_rows
    ]

    summary = summarize_latencies(
        latencies
    )

    first_latencies = [
        row["latency_ms"]
        for row in method_rows
        if row["order_position"] == 1
    ]

    second_latencies = [
        row["latency_ms"]
        for row in method_rows
        if row["order_position"] == 2
    ]

    summary[
        "first_count"
    ] = len(
        first_latencies
    )

    summary[
        "second_count"
    ] = len(
        second_latencies
    )

    summary[
        "mean_when_first_ms"
    ] = statistics.mean(
        first_latencies
    )

    summary[
        "mean_when_second_ms"
    ] = statistics.mean(
        second_latencies
    )

    summary[
        "second_minus_first_ms"
    ] = (
        summary[
            "mean_when_second_ms"
        ]
        - summary[
            "mean_when_first_ms"
        ]
    )

    return summary


def calculate_paired_differences(
    rows,
):
    """
    For each:
        repeat + image

    compute:

        Patch latency - Global latency

    Negative:
        Patch faster

    Positive:
        Global faster
    """

    pairs = {}

    for row in rows:

        key = (
            row["repeat"],
            row["image_index"],
        )

        if key not in pairs:
            pairs[key] = {}

        pairs[key][
            row["method"]
        ] = row[
            "latency_ms"
        ]

    differences = []

    for key, values in pairs.items():

        if (
            "global" not in values
            or "patch_layer3" not in values
        ):
            raise RuntimeError(
                f"Incomplete pair: {key}"
            )

        difference = (
            values["patch_layer3"]
            - values["global"]
        )

        differences.append(
            difference
        )

    return differences


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "EXP022 - Interleaved "
        "Balanced-Order E2E Benchmark"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # CUDA
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required."
        )

    device = torch.device(
        "cuda"
    )

    torch.backends.cudnn.benchmark = (
        True
    )

    print(
        f"\nTorch: "
        f"{torch.__version__}"
    )

    print(
        f"CUDA build: "
        f"{torch.version.cuda}"
    )

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    # --------------------------------------------------------
    # Load EXP018 split
    # --------------------------------------------------------

    exp018 = torch.load(
        "results/exp018/"
        "threshold_calibration.pt",
        map_location="cpu",
    )

    reference_paths = [
        Path(path)
        for path
        in exp018[
            "reference_paths"
        ]
    ]

    test_paths = [
        Path(path)
        for path
        in exp018[
            "test_paths"
        ]
    ]

    test_labels = (
        exp018[
            "test_labels"
        ]
        .long()
        .tolist()
    )

    print(
        f"\nReference images: "
        f"{len(reference_paths)}"
    )

    print(
        f"Test images: "
        f"{len(test_paths)}"
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    weights = (
        ResNet18_Weights.DEFAULT
    )

    preprocess = (
        weights.transforms()
    )

    # Only reference tensors are precomputed.
    # Test images remain disk -> decode -> preprocess
    # during every measured E2E request.
    reference_tensors, _ = (
        preprocess_paths(
            reference_paths,
            preprocess,
            "reference",
        )
    )

    # --------------------------------------------------------
    # Build BOTH methods
    # --------------------------------------------------------
    #
    # Important:
    # Both methods remain resident on GPU so that
    # they can be measured immediately next to
    # each other.
    #
    # This experiment is NOT a peak-VRAM benchmark.
    # Peak VRAM was already measured separately.
    # --------------------------------------------------------

    print(
        "\nBuilding Global..."
    )

    global_model, _ = (
        build_global_model(
            weights,
            device,
        )
    )

    global_center, _ = (
        build_global_center(
            global_model,
            reference_tensors,
            device,
        )
    )

    print(
        "\nBuilding Patch-layer3..."
    )

    patch_model, _ = (
        build_patch_model(
            weights,
            device,
        )
    )

    (
        patch_coreset,
        _,
        patch_full_memory_shape,
    ) = build_patch_reference(
        patch_model,
        reference_tensors,
        device,
    )

    print(
        "\nPatch full memory shape:"
    )

    print(
        patch_full_memory_shape
    )

    # --------------------------------------------------------
    # Score functions
    # --------------------------------------------------------

    global_score_function = (
        lambda tensor:
            global_score(
                global_model,
                global_center,
                tensor,
                device,
            )
    )

    patch_score_function = (
        lambda tensor:
            patch_score(
                patch_model,
                patch_coreset,
                tensor,
                device,
            )
    )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    warmup_interleaved(
        test_paths=
            test_paths,

        preprocess=
            preprocess,

        global_score_function=
            global_score_function,

        patch_score_function=
            patch_score_function,

        device=
            device,
    )

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    (
        rows,
        scores,
    ) = benchmark_interleaved(
        test_paths=
            test_paths,

        preprocess=
            preprocess,

        global_score_function=
            global_score_function,

        patch_score_function=
            patch_score_function,

        device=
            device,
    )

    # --------------------------------------------------------
    # Accuracy regression
    # --------------------------------------------------------

    global_accuracy = (
        calculate_accuracy(
            test_labels,
            scores["global"],
        )
    )

    patch_accuracy = (
        calculate_accuracy(
            test_labels,
            scores["patch_layer3"],
        )
    )

    # --------------------------------------------------------
    # Latency summaries
    # --------------------------------------------------------

    global_summary = (
        analyze_method(
            rows,
            "global",
        )
    )

    patch_summary = (
        analyze_method(
            rows,
            "patch_layer3",
        )
    )

    paired_differences = (
        calculate_paired_differences(
            rows
        )
    )

    paired_mean = (
        statistics.mean(
            paired_differences
        )
    )

    paired_median = (
        statistics.median(
            paired_differences
        )
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "INTERLEAVED RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        "\nGlobal:"
    )

    print(
        f"  AUROC: "
        f"{global_accuracy['auroc']:.6f}"
    )

    print(
        f"  AP: "
        f"{global_accuracy['ap']:.6f}"
    )

    print(
        f"  Mean: "
        f"{global_summary['mean_latency_ms']:.3f} ms"
    )

    print(
        f"  P95: "
        f"{global_summary['p95_latency_ms']:.3f} ms"
    )

    print(
        f"  Throughput: "
        f"{global_summary['throughput_images_s']:.3f} img/s"
    )

    print(
        f"  Mean when first: "
        f"{global_summary['mean_when_first_ms']:.3f} ms"
    )

    print(
        f"  Mean when second: "
        f"{global_summary['mean_when_second_ms']:.3f} ms"
    )

    print(
        f"  Second - first: "
        f"{global_summary['second_minus_first_ms']:.3f} ms"
    )

    print(
        "\nPatch-layer3:"
    )

    print(
        f"  AUROC: "
        f"{patch_accuracy['auroc']:.6f}"
    )

    print(
        f"  AP: "
        f"{patch_accuracy['ap']:.6f}"
    )

    print(
        f"  Mean: "
        f"{patch_summary['mean_latency_ms']:.3f} ms"
    )

    print(
        f"  P95: "
        f"{patch_summary['p95_latency_ms']:.3f} ms"
    )

    print(
        f"  Throughput: "
        f"{patch_summary['throughput_images_s']:.3f} img/s"
    )

    print(
        f"  Mean when first: "
        f"{patch_summary['mean_when_first_ms']:.3f} ms"
    )

    print(
        f"  Mean when second: "
        f"{patch_summary['mean_when_second_ms']:.3f} ms"
    )

    print(
        f"  Second - first: "
        f"{patch_summary['second_minus_first_ms']:.3f} ms"
    )

    print(
        "\nPaired comparison:"
    )

    print(
        "  Definition:"
    )

    print(
        "    Patch latency - Global latency"
    )

    print(
        f"  Mean paired difference: "
        f"{paired_mean:.3f} ms"
    )

    print(
        f"  Median paired difference: "
        f"{paired_median:.3f} ms"
    )

    print(
        "\nInterpretation:"
    )

    print(
        "  negative = Patch faster"
    )

    print(
        "  positive = Global faster"
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    output_dir = Path(
        "results/exp022"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples_csv = (
        output_dir
        / "engineering_interleaved_e2e_samples.csv"
    )

    summary_csv = (
        output_dir
        / "engineering_interleaved_e2e_summary.csv"
    )

    with open(
        samples_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    summary_row = {
        "global_auroc":
            global_accuracy["auroc"],

        "global_ap":
            global_accuracy["ap"],

        "global_mean_ms":
            global_summary[
                "mean_latency_ms"
            ],

        "global_p95_ms":
            global_summary[
                "p95_latency_ms"
            ],

        "global_throughput_images_s":
            global_summary[
                "throughput_images_s"
            ],

        "global_first_count":
            global_summary[
                "first_count"
            ],

        "global_second_count":
            global_summary[
                "second_count"
            ],

        "global_mean_when_first_ms":
            global_summary[
                "mean_when_first_ms"
            ],

        "global_mean_when_second_ms":
            global_summary[
                "mean_when_second_ms"
            ],

        "global_second_minus_first_ms":
            global_summary[
                "second_minus_first_ms"
            ],

        "patch_auroc":
            patch_accuracy["auroc"],

        "patch_ap":
            patch_accuracy["ap"],

        "patch_mean_ms":
            patch_summary[
                "mean_latency_ms"
            ],

        "patch_p95_ms":
            patch_summary[
                "p95_latency_ms"
            ],

        "patch_throughput_images_s":
            patch_summary[
                "throughput_images_s"
            ],

        "patch_first_count":
            patch_summary[
                "first_count"
            ],

        "patch_second_count":
            patch_summary[
                "second_count"
            ],

        "patch_mean_when_first_ms":
            patch_summary[
                "mean_when_first_ms"
            ],

        "patch_mean_when_second_ms":
            patch_summary[
                "mean_when_second_ms"
            ],

        "patch_second_minus_first_ms":
            patch_summary[
                "second_minus_first_ms"
            ],

        "paired_patch_minus_global_mean_ms":
            paired_mean,

        "paired_patch_minus_global_median_ms":
            paired_median,
    }

    with open(
        summary_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summary_row.keys()
            ),
        )

        writer.writeheader()

        writer.writerow(
            summary_row
        )

    print(
        "\nFiles:"
    )

    print(
        samples_csv.resolve()
    )

    print(
        summary_csv.resolve()
    )

    print(
        "\nEXP022 Active Modification 3 completed."
    )


if __name__ == "__main__":
    main()