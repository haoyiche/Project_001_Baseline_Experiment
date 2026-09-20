from pathlib import Path
import csv
import statistics
import time

from PIL import Image

import torch
from torchvision.models import (
    ResNet18_Weights,
)

from src.experiments.engineering_benchmark import (
    build_global_model,
    build_patch_model,
    build_global_center,
    build_patch_reference,
    preprocess_paths,
    synchronize,
)


# ============================================================
# Configuration
# ============================================================

BENCHMARK_REPEATS = 5


# ============================================================
# GPU-Only Scoring
# ============================================================

@torch.inference_mode()
def global_score_gpu(
    model,
    center,
    gpu_tensor,
):

    # gpu_tensor:
    # [1,3,224,224]

    feature = model(
        gpu_tensor
    )[0]

    score = (
        torch.linalg.vector_norm(
            feature - center,
            ord=2,
        )
    )

    return score


@torch.inference_mode()
def patch_score_gpu(
    model,
    coreset,
    gpu_tensor,
):

    # [1,3,224,224]
    # ->
    # [1,256,14,14]
    feature_map = model(
        gpu_tensor
    )

    # [1,256,14,14]
    # ->
    # [196,256]
    patches = (
        feature_map
        .permute(
            0,
            2,
            3,
            1,
        )
        .reshape(
            -1,
            feature_map.shape[1],
        )
    )

    # [196,256]
    # x
    # [100,256]
    #
    # ->
    # [196,100]
    distances = torch.cdist(
        patches,
        coreset,
        p=2,
    )

    # [196,100]
    # ->
    # [196]
    patch_scores = (
        distances.min(
            dim=1
        ).values
    )

    # [196]
    # ->
    # scalar
    score = (
        patch_scores.max()
    )

    return score


# ============================================================
# Statistics
# ============================================================

def p95(
    values,
):

    tensor = torch.tensor(
        values,
        dtype=torch.float64,
    )

    return torch.quantile(
        tensor,
        0.95,
    ).item()


def summarize_stage(
    values,
):

    return {
        "mean_ms":
            statistics.mean(
                values
            ),

        "median_ms":
            statistics.median(
                values
            ),

        "p95_ms":
            p95(
                values
            ),

        "min_ms":
            min(
                values
            ),

        "max_ms":
            max(
                values
            ),
    }


# ============================================================
# Stage Profiler
# ============================================================

def profile_method(
    name,
    test_paths,
    preprocess,
    device,
    gpu_score_function,
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"Stage Profile: {name}"
    )

    print(
        "=" * 70
    )

    rows = []

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print(
        "\nWarm-up..."
    )

    for index in range(20):

        path = test_paths[
            index % len(test_paths)
        ]

        with Image.open(
            path
        ) as image:

            rgb = image.convert(
                "RGB"
            )

        tensor = preprocess(
            rgb
        )

        gpu_tensor = (
            tensor
            .unsqueeze(0)
            .to(device)
        )

        gpu_score_function(
            gpu_tensor
        )

    synchronize(
        device
    )

    # --------------------------------------------------------
    # Repeated Profiling
    # --------------------------------------------------------

    for repeat in range(
        BENCHMARK_REPEATS
    ):

        print(
            f"  repeat "
            f"{repeat + 1}/"
            f"{BENCHMARK_REPEATS}"
        )

        for image_index, path in enumerate(
            test_paths
        ):

            # =================================================
            # Stage 1:
            # PNG open + decode + RGB conversion
            # =================================================

            start = (
                time.perf_counter()
            )

            with Image.open(
                path
            ) as image:

                rgb = image.convert(
                    "RGB"
                )

            decode_ms = (
                time.perf_counter()
                - start
            ) * 1000.0

            # =================================================
            # Stage 2:
            # PIL RGB -> normalized CPU tensor
            # =================================================

            start = (
                time.perf_counter()
            )

            cpu_tensor = preprocess(
                rgb
            )

            preprocess_ms = (
                time.perf_counter()
                - start
            ) * 1000.0

            # =================================================
            # Stage 3:
            # CPU tensor -> GPU tensor
            # =================================================

            synchronize(
                device
            )

            start = (
                time.perf_counter()
            )

            gpu_tensor = (
                cpu_tensor
                .unsqueeze(0)
                .to(device)
            )

            synchronize(
                device
            )

            h2d_ms = (
                time.perf_counter()
                - start
            ) * 1000.0

            # =================================================
            # Stage 4:
            # GPU backbone + anomaly scoring
            # =================================================

            synchronize(
                device
            )

            start = (
                time.perf_counter()
            )

            score_tensor = (
                gpu_score_function(
                    gpu_tensor
                )
            )

            synchronize(
                device
            )

            model_score_ms = (
                time.perf_counter()
                - start
            ) * 1000.0

            score = (
                score_tensor.item()
            )

            stage_sum_ms = (
                decode_ms
                + preprocess_ms
                + h2d_ms
                + model_score_ms
            )

            rows.append(
                {
                    "method":
                        name,

                    "repeat":
                        repeat,

                    "image_index":
                        image_index,

                    "path":
                        str(path),

                    "decode_ms":
                        decode_ms,

                    "preprocess_ms":
                        preprocess_ms,

                    "h2d_ms":
                        h2d_ms,

                    "model_score_ms":
                        model_score_ms,

                    "stage_sum_ms":
                        stage_sum_ms,

                    "score":
                        score,
                }
            )

    return rows


# ============================================================
# Summarize Profile
# ============================================================

def summarize_method(
    name,
    rows,
):

    stage_names = [
        "decode_ms",
        "preprocess_ms",
        "h2d_ms",
        "model_score_ms",
    ]

    summaries = {}

    print(
        "\n"
        + "-" * 70
    )

    print(
        f"Stage Summary: {name}"
    )

    print(
        "-" * 70
    )

    mean_sum = 0.0

    for stage in stage_names:

        values = [
            row[stage]
            for row in rows
        ]

        result = (
            summarize_stage(
                values
            )
        )

        summaries[
            stage
        ] = result

        mean_sum += (
            result[
                "mean_ms"
            ]
        )

    for stage in stage_names:

        result = (
            summaries[
                stage
            ]
        )

        share_percent = (
            result[
                "mean_ms"
            ]
            / mean_sum
            * 100.0
        )

        result[
            "share_percent"
        ] = (
            share_percent
        )

        print(
            f"\n{stage}"
        )

        print(
            f"  mean   = "
            f"{result['mean_ms']:.3f} ms"
        )

        print(
            f"  P95    = "
            f"{result['p95_ms']:.3f} ms"
        )

        print(
            f"  share  = "
            f"{share_percent:.2f}%"
        )

    print(
        f"\nMean stage sum: "
        f"{mean_sum:.3f} ms"
    )

    return summaries


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "EXP022 - End-to-End Stage Profiling"
    )

    print(
        "=" * 70
    )

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

    # ========================================================
    # Load EXP018 Split
    # ========================================================

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

    print(
        f"\nReference images: "
        f"{len(reference_paths)}"
    )

    print(
        f"Test images: "
        f"{len(test_paths)}"
    )

    # ========================================================
    # Preprocessing Definition
    # ========================================================

    weights = (
        ResNet18_Weights.DEFAULT
    )

    preprocess = (
        weights.transforms()
    )

    # Reference tensors are only needed
    # to rebuild exactly the same
    # normal references.
    reference_tensors, _ = (
        preprocess_paths(
            reference_paths,
            preprocess,
            "reference",
        )
    )

    # ========================================================
    # GLOBAL
    # ========================================================

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

    global_rows = (
        profile_method(
            name=
                "global",

            test_paths=
                test_paths,

            preprocess=
                preprocess,

            device=
                device,

            gpu_score_function=
                lambda gpu_tensor:
                    global_score_gpu(
                        global_model,
                        global_center,
                        gpu_tensor,
                    ),
        )
    )

    global_summary = (
        summarize_method(
            "global",
            global_rows,
        )
    )

    del global_model
    del global_center

    torch.cuda.empty_cache()

    synchronize(
        device
    )

    # ========================================================
    # PATCH
    # ========================================================

    patch_model, _ = (
        build_patch_model(
            weights,
            device,
        )
    )

    (
        patch_coreset,
        _,
        _,
    ) = build_patch_reference(
        patch_model,
        reference_tensors,
        device,
    )

    patch_rows = (
        profile_method(
            name=
                "patch_layer3",

            test_paths=
                test_paths,

            preprocess=
                preprocess,

            device=
                device,

            gpu_score_function=
                lambda gpu_tensor:
                    patch_score_gpu(
                        patch_model,
                        patch_coreset,
                        gpu_tensor,
                    ),
        )
    )

    patch_summary = (
        summarize_method(
            "patch_layer3",
            patch_rows,
        )
    )

    # ========================================================
    # Save Results
    # ========================================================

    output_dir = Path(
        "results/exp022"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples_csv = (
        output_dir
        / "engineering_stage_profile_samples.csv"
    )

    summary_csv = (
        output_dir
        / "engineering_stage_profile_summary.csv"
    )

    all_rows = (
        global_rows
        + patch_rows
    )

    with open(
        samples_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=
                list(
                    all_rows[0]
                    .keys()
                ),
        )

        writer.writeheader()

        writer.writerows(
            all_rows
        )

    summary_rows = []

    for method_name, summary in [
        (
            "global",
            global_summary,
        ),
        (
            "patch_layer3",
            patch_summary,
        ),
    ]:

        for stage_name, result in (
            summary.items()
        ):

            summary_rows.append(
                {
                    "method":
                        method_name,

                    "stage":
                        stage_name,

                    "mean_ms":
                        result[
                            "mean_ms"
                        ],

                    "median_ms":
                        result[
                            "median_ms"
                        ],

                    "p95_ms":
                        result[
                            "p95_ms"
                        ],

                    "min_ms":
                        result[
                            "min_ms"
                        ],

                    "max_ms":
                        result[
                            "max_ms"
                        ],

                    "share_percent":
                        result[
                            "share_percent"
                        ],
                }
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

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FILES"
    )

    print(
        "=" * 70
    )

    print(
        samples_csv.resolve()
    )

    print(
        summary_csv.resolve()
    )

    print(
        "\nStage profiling completed."
    )


if __name__ == "__main__":

    main()