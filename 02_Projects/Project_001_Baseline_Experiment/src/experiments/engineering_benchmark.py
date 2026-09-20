from pathlib import Path
import csv
import statistics
import time

from PIL import Image

import torch
import torch.nn as nn

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)

from src.experiments.coreset_stability_repeated_seed import (
    build_coreset,
)


# ============================================================
# Configuration
# ============================================================

WARMUP_ITERS = 20
BENCHMARK_REPEATS = 5

PATCH_SEED = 42
CANDIDATE_SIZE = 5000
CORESET_SIZE = 100

OFFLINE_BATCH_SIZE = 16


# ============================================================
# Patch Layer3 Backbone
# ============================================================

class ResNet18Layer3(nn.Module):

    def __init__(
        self,
        base_model,
    ):

        super().__init__()

        self.conv1 = (
            base_model.conv1
        )

        self.bn1 = (
            base_model.bn1
        )

        self.relu = (
            base_model.relu
        )

        self.maxpool = (
            base_model.maxpool
        )

        self.layer1 = (
            base_model.layer1
        )

        self.layer2 = (
            base_model.layer2
        )

        self.layer3 = (
            base_model.layer3
        )

    def forward(
        self,
        x,
    ):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


# ============================================================
# CUDA / Statistics Utilities
# ============================================================

def synchronize(
    device,
):

    if device.type == "cuda":

        torch.cuda.synchronize()


def percentile(
    values,
    q,
):

    tensor = torch.tensor(
        values,
        dtype=torch.float64,
    )

    return torch.quantile(
        tensor,
        q,
    ).item()


def summarize_latencies(
    latencies_ms,
):

    total_seconds = (
        sum(latencies_ms)
        / 1000.0
    )

    throughput = (
        len(latencies_ms)
        / total_seconds
    )

    return {
        "mean_latency_ms":
            statistics.mean(
                latencies_ms
            ),

        "median_latency_ms":
            statistics.median(
                latencies_ms
            ),

        "p95_latency_ms":
            percentile(
                latencies_ms,
                0.95,
            ),

        "min_latency_ms":
            min(
                latencies_ms
            ),

        "max_latency_ms":
            max(
                latencies_ms
            ),

        "throughput_images_s":
            throughput,
    }


def representation_bytes(
    tensor,
):

    return (
        tensor.numel()
        * tensor.element_size()
    )


# ============================================================
# Shared Image Preprocessing
# ============================================================

def preprocess_paths(
    image_paths,
    preprocess,
    name,
):

    tensors = []
    times_ms = []

    print(
        f"\nPreprocessing {name}..."
    )

    for index, path in enumerate(
        image_paths,
        start=1,
    ):

        start = (
            time.perf_counter()
        )

        with Image.open(
            path
        ) as image:

            image = (
                image.convert("RGB")
            )

            tensor = preprocess(
                image
            )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        tensors.append(
            tensor
        )

        times_ms.append(
            elapsed_ms
        )

        if (
            index % 25 == 0
            or index
            == len(image_paths)
        ):

            print(
                f"  {name}: "
                f"{index}/"
                f"{len(image_paths)}"
            )

    return (
        tensors,
        times_ms,
    )


# ============================================================
# Model Startup
# ============================================================

def build_global_model(
    weights,
    device,
):

    torch.cuda.empty_cache()

    synchronize(
        device
    )

    start = (
        time.perf_counter()
    )

    model = resnet18(
        weights=weights
    )

    # Remove ImageNet classifier.
    # Output becomes [B, 512].
    model.fc = (
        nn.Identity()
    )

    model.eval()

    model.to(
        device
    )

    synchronize(
        device
    )

    startup_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    return (
        model,
        startup_ms,
    )


def build_patch_model(
    weights,
    device,
):

    torch.cuda.empty_cache()

    synchronize(
        device
    )

    start = (
        time.perf_counter()
    )

    base_model = resnet18(
        weights=weights
    )

    model = ResNet18Layer3(
        base_model
    )

    del base_model

    model.eval()

    model.to(
        device
    )

    synchronize(
        device
    )

    startup_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    return (
        model,
        startup_ms,
    )


# ============================================================
# Global Offline Preparation
# ============================================================

@torch.inference_mode()
def build_global_center(
    model,
    reference_tensors,
    device,
):

    synchronize(
        device
    )

    start = (
        time.perf_counter()
    )

    feature_parts = []

    for start_index in range(
        0,
        len(reference_tensors),
        OFFLINE_BATCH_SIZE,
    ):

        batch = torch.stack(
            reference_tensors[
                start_index:
                start_index
                + OFFLINE_BATCH_SIZE
            ]
        ).to(
            device
        )

        features = model(
            batch
        )

        feature_parts.append(
            features.cpu()
        )

    synchronize(
        device
    )

    all_features = torch.cat(
        feature_parts,
        dim=0,
    )

    # [N, 512]
    # -> [512]
    center_cpu = (
        all_features.mean(
            dim=0
        )
    )

    center = (
        center_cpu.to(
            device
        )
    )

    synchronize(
        device
    )

    elapsed_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    return (
        center,
        elapsed_ms,
    )


# ============================================================
# Patch Offline Preparation
# ============================================================

@torch.inference_mode()
def build_patch_reference(
    model,
    reference_tensors,
    device,
):

    synchronize(
        device
    )

    start = (
        time.perf_counter()
    )

    memory_parts = []

    for start_index in range(
        0,
        len(reference_tensors),
        OFFLINE_BATCH_SIZE,
    ):

        batch = torch.stack(
            reference_tensors[
                start_index:
                start_index
                + OFFLINE_BATCH_SIZE
            ]
        ).to(
            device
        )

        # Example:
        #
        # [B, 3, 224, 224]
        # ->
        # [B, 256, 14, 14]
        feature_map = model(
            batch
        )

        # [B, 256, 14, 14]
        # ->
        # [B, 14, 14, 256]
        patches = (
            feature_map.permute(
                0,
                2,
                3,
                1,
            )
        )

        # [B, 14, 14, 256]
        # ->
        # [B*196, 256]
        patches = (
            patches.reshape(
                -1,
                feature_map.shape[1],
            )
        )

        memory_parts.append(
            patches.cpu()
        )

    synchronize(
        device
    )

    # 167 images
    # * 196 patches/image
    # = 32732 patches
    #
    # [32732, 256]
    full_memory = torch.cat(
        memory_parts,
        dim=0,
    )

    print(
        "\nPatch full memory:"
    )

    print(
        tuple(
            full_memory.shape
        )
    )

    coreset_result = (
        build_coreset(
            full_memory_bank=
                full_memory,

            seed=
                PATCH_SEED,

            candidate_size=
                CANDIDATE_SIZE,

            coreset_size=
                CORESET_SIZE,
        )
    )

    # [100, 256]
    coreset_cpu = (
        coreset_result[
            "coreset_bank"
        ]
    )

    coreset = (
        coreset_cpu.to(
            device
        )
    )

    synchronize(
        device
    )

    elapsed_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    return (
        coreset,
        elapsed_ms,
        full_memory.shape,
    )


# ============================================================
# Global Online Scoring
# ============================================================

@torch.inference_mode()
def global_score(
    model,
    center,
    cpu_tensor,
    device,
):

    # [3,224,224]
    # ->
    # [1,3,224,224]
    x = (
        cpu_tensor
        .unsqueeze(0)
        .to(device)
    )

    # [1,512]
    # ->
    # [512]
    feature = (
        model(x)[0]
    )

    # L2 distance from
    # normal center.
    score = (
        torch.linalg.vector_norm(
            feature
            - center,

            ord=2,
        )
    )

    return score


# ============================================================
# Patch Online Scoring
# ============================================================

@torch.inference_mode()
def patch_score(
    model,
    coreset,
    cpu_tensor,
    device,
):

    # [3,224,224]
    # ->
    # [1,3,224,224]
    x = (
        cpu_tensor
        .unsqueeze(0)
        .to(device)
    )

    # [1,256,14,14]
    feature_map = model(
        x
    )

    # [1,256,14,14]
    # ->
    # [1,14,14,256]
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

    # patches:
    # [196,256]
    #
    # coreset:
    # [100,256]
    #
    # output:
    # [196,100]
    distances = (
        torch.cdist(
            patches,
            coreset,
            p=2,
        )
    )

    # For every query patch,
    # find nearest normal prototype.
    #
    # [196,100]
    # ->
    # [196]
    patch_scores = (
        distances.min(
            dim=1
        ).values
    )

    # MAX aggregation:
    #
    # [196]
    # ->
    # scalar
    score = (
        patch_scores.max()
    )

    return score


# ============================================================
# Model-Only Benchmark
# ============================================================

def benchmark_method(
    name,
    test_tensors,
    test_paths,
    score_function,
    device,
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"Model-Only Benchmark: "
        f"{name}"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print(
        f"\nWarm-up: "
        f"{WARMUP_ITERS} iterations"
    )

    for _ in range(
        WARMUP_ITERS
    ):

        score_function(
            test_tensors[0]
        )

    synchronize(
        device
    )

    # --------------------------------------------------------
    # Reset Peak VRAM
    # --------------------------------------------------------

    if device.type == "cuda":

        torch.cuda.reset_peak_memory_stats(
            device
        )

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    latency_rows = []

    first_repeat_scores = []

    for repeat in range(
        BENCHMARK_REPEATS
    ):

        print(
            f"  repeat "
            f"{repeat + 1}/"
            f"{BENCHMARK_REPEATS}"
        )

        for image_index, (
            tensor,
            path,
        ) in enumerate(
            zip(
                test_tensors,
                test_paths,
            )
        ):

            # Make sure previous CUDA work
            # has completed.
            synchronize(
                device
            )

            start = (
                time.perf_counter()
            )

            score_tensor = (
                score_function(
                    tensor
                )
            )

            # Wait until current GPU work
            # is actually finished.
            synchronize(
                device
            )

            elapsed_ms = (
                time.perf_counter()
                - start
            ) * 1000.0

            score = (
                score_tensor.item()
            )

            # Accuracy only needs
            # one copy of each image.
            if repeat == 0:

                first_repeat_scores.append(
                    score
                )

            latency_rows.append(
                {
                    "method":
                        name,

                    "measurement_scope":
                        "model_only",

                    "repeat":
                        repeat,

                    "image_index":
                        image_index,

                    "path":
                        str(path),

                    "latency_ms":
                        elapsed_ms,

                    "score":
                        score,
                }
            )

    latencies = [
        row[
            "latency_ms"
        ]
        for row
        in latency_rows
    ]

    summary = (
        summarize_latencies(
            latencies
        )
    )

    if device.type == "cuda":

        peak_vram_mib = (
            torch.cuda
            .max_memory_allocated(
                device
            )
            / (1024 ** 2)
        )

    else:

        peak_vram_mib = (
            float("nan")
        )

    summary[
        "peak_allocated_vram_mib"
    ] = (
        peak_vram_mib
    )

    print(
        f"\nMean latency: "
        f"{summary['mean_latency_ms']:.3f} ms"
    )

    print(
        f"P95 latency: "
        f"{summary['p95_latency_ms']:.3f} ms"
    )

    print(
        f"Throughput: "
        f"{summary['throughput_images_s']:.3f} "
        f"images/s"
    )

    print(
        f"Peak allocated VRAM: "
        f"{peak_vram_mib:.2f} MiB"
    )

    return (
        summary,
        latency_rows,
        first_repeat_scores,
    )


# ============================================================
# End-to-End Benchmark
# ============================================================

def benchmark_end_to_end(
    name,
    test_paths,
    preprocess,
    score_function,
    device,
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"End-to-End Benchmark: "
        f"{name}"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print(
        f"\nWarm-up: "
        f"{WARMUP_ITERS} iterations"
    )

    for index in range(
        WARMUP_ITERS
    ):

        path = (
            test_paths[
                index
                % len(test_paths)
            ]
        )

        with Image.open(
            path
        ) as image:

            image = (
                image.convert(
                    "RGB"
                )
            )

            tensor = (
                preprocess(
                    image
                )
            )

        score_function(
            tensor
        )

    synchronize(
        device
    )

    latency_rows = []

    # --------------------------------------------------------
    # Repeated E2E Benchmark
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

            # Clear previous GPU work.
            synchronize(
                device
            )

            start = (
                time.perf_counter()
            )

            # --------------------------------------------
            # 1. Disk PNG -> PIL
            # --------------------------------------------

            with Image.open(
                path
            ) as image:

                image = (
                    image.convert(
                        "RGB"
                    )
                )

                # ----------------------------------------
                # 2. PIL -> normalized CPU tensor
                # ----------------------------------------

                tensor = (
                    preprocess(
                        image
                    )
                )

            # --------------------------------------------
            # 3. CPU tensor -> GPU
            # 4. Backbone
            # 5. Anomaly scoring
            # --------------------------------------------

            score_tensor = (
                score_function(
                    tensor
                )
            )

            synchronize(
                device
            )

            elapsed_ms = (
                time.perf_counter()
                - start
            ) * 1000.0

            score = (
                score_tensor.item()
            )

            latency_rows.append(
                {
                    "method":
                        name,

                    "measurement_scope":
                        "end_to_end",

                    "repeat":
                        repeat,

                    "image_index":
                        image_index,

                    "path":
                        str(path),

                    "latency_ms":
                        elapsed_ms,

                    "score":
                        score,
                }
            )

    latencies = [
        row[
            "latency_ms"
        ]
        for row
        in latency_rows
    ]

    summary = (
        summarize_latencies(
            latencies
        )
    )

    print(
        f"\nE2E mean latency: "
        f"{summary['mean_latency_ms']:.3f} ms"
    )

    print(
        f"E2E P95 latency: "
        f"{summary['p95_latency_ms']:.3f} ms"
    )

    print(
        f"E2E throughput: "
        f"{summary['throughput_images_s']:.3f} "
        f"images/s"
    )

    return (
        summary,
        latency_rows,
    )


# ============================================================
# Accuracy
# ============================================================

def calculate_accuracy(
    labels,
    scores,
):

    auroc = (
        roc_auc_score(
            labels,
            scores,
        )
    )

    ap = (
        average_precision_score(
            labels,
            scores,
        )
    )

    return {
        "auroc":
            float(
                auroc
            ),

        "ap":
            float(
                ap
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "EXP022 - Engineering Evaluation"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # 0. CUDA Check
    # ========================================================

    if not torch.cuda.is_available():

        raise RuntimeError(
            "EXP022 requires CUDA."
        )

    device = torch.device(
        "cuda"
    )

    # Input shape is fixed:
    # [1,3,224,224].
    #
    # Let cuDNN select
    # efficient algorithms.
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

    properties = (
        torch.cuda
        .get_device_properties(
            device
        )
    )

    print(
        f"GPU total VRAM: "
        f"{properties.total_memory / 1024**3:.2f} "
        f"GiB"
    )

    # ========================================================
    # 1. Load EXP018 Split
    # ========================================================

    exp018 = torch.load(
        "results/exp018/"
        "threshold_calibration.pt",

        map_location="cpu",
    )

    reference_paths = [
        Path(
            path
        )
        for path
        in exp018[
            "reference_paths"
        ]
    ]

    test_paths = [
        Path(
            path
        )
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

    # ========================================================
    # 2. Shared Preprocessing
    # ========================================================

    weights = (
        ResNet18_Weights.DEFAULT
    )

    preprocess = (
        weights.transforms()
    )

    reference_tensors, (
        reference_preprocess_times
    ) = preprocess_paths(
        reference_paths,
        preprocess,
        "reference",
    )

    test_tensors, (
        test_preprocess_times
    ) = preprocess_paths(
        test_paths,
        preprocess,
        "test",
    )

    preprocess_summary = (
        summarize_latencies(
            test_preprocess_times
        )
    )

    print(
        "\nShared test preprocessing:"
    )

    print(
        f"  mean = "
        f"{preprocess_summary['mean_latency_ms']:.3f} ms"
    )

    print(
        f"  P95  = "
        f"{preprocess_summary['p95_latency_ms']:.3f} ms"
    )

    # ========================================================
    # 3. GLOBAL Model
    # ========================================================

    global_model, (
        global_startup_ms
    ) = build_global_model(
        weights,
        device,
    )

    global_center, (
        global_offline_ms
    ) = build_global_center(
        global_model,
        reference_tensors,
        device,
    )

    global_rep_bytes = (
        representation_bytes(
            global_center
        )
    )

    # --------------------------------------------------------
    # GLOBAL Model-Only Benchmark
    # --------------------------------------------------------

    (
        global_benchmark,
        global_latency_rows,
        global_scores,
    ) = benchmark_method(
        name=
            "global",

        test_tensors=
            test_tensors,

        test_paths=
            test_paths,

        score_function=
            lambda tensor:
                global_score(
                    global_model,
                    global_center,
                    tensor,
                    device,
                ),

        device=
            device,
    )

    # --------------------------------------------------------
    # GLOBAL Accuracy
    # --------------------------------------------------------

    global_accuracy = (
        calculate_accuracy(
            test_labels,
            global_scores,
        )
    )

    print(
        f"\nGlobal AUROC: "
        f"{global_accuracy['auroc']:.6f}"
    )

    print(
        f"Global AP: "
        f"{global_accuracy['ap']:.6f}"
    )

    global_expected_auroc = (
        0.976984
    )

    global_delta = abs(
        global_accuracy[
            "auroc"
        ]
        - global_expected_auroc
    )

    print(
        f"Global AUROC delta "
        f"vs EXP018: "
        f"{global_delta:.8f}"
    )

    # --------------------------------------------------------
    # GLOBAL End-to-End Benchmark
    # --------------------------------------------------------

    (
        global_e2e,
        global_e2e_rows,
    ) = benchmark_end_to_end(
        name=
            "global",

        test_paths=
            test_paths,

        preprocess=
            preprocess,

        score_function=
            lambda tensor:
                global_score(
                    global_model,
                    global_center,
                    tensor,
                    device,
                ),

        device=
            device,
    )

    # --------------------------------------------------------
    # Free GLOBAL
    # --------------------------------------------------------

    del global_model
    del global_center

    torch.cuda.empty_cache()

    synchronize(
        device
    )

    # ========================================================
    # 4. PATCH-LAYER3 Model
    # ========================================================

    patch_model, (
        patch_startup_ms
    ) = build_patch_model(
        weights,
        device,
    )

    (
        patch_coreset,
        patch_offline_ms,
        patch_full_memory_shape,
    ) = build_patch_reference(
        patch_model,
        reference_tensors,
        device,
    )

    patch_rep_bytes = (
        representation_bytes(
            patch_coreset
        )
    )

    # --------------------------------------------------------
    # PATCH Model-Only Benchmark
    # --------------------------------------------------------

    (
        patch_benchmark,
        patch_latency_rows,
        patch_scores,
    ) = benchmark_method(
        name=
            "patch_layer3",

        test_tensors=
            test_tensors,

        test_paths=
            test_paths,

        score_function=
            lambda tensor:
                patch_score(
                    patch_model,
                    patch_coreset,
                    tensor,
                    device,
                ),

        device=
            device,
    )

    # --------------------------------------------------------
    # PATCH Accuracy
    # --------------------------------------------------------

    patch_accuracy = (
        calculate_accuracy(
            test_labels,
            patch_scores,
        )
    )

    print(
        f"\nPatch AUROC: "
        f"{patch_accuracy['auroc']:.6f}"
    )

    print(
        f"Patch AP: "
        f"{patch_accuracy['ap']:.6f}"
    )

    patch_expected_auroc = (
        1.000000
    )

    patch_delta = abs(
        patch_accuracy[
            "auroc"
        ]
        - patch_expected_auroc
    )

    print(
        f"Patch AUROC delta "
        f"vs EXP021 seed42: "
        f"{patch_delta:.8f}"
    )

    # --------------------------------------------------------
    # PATCH End-to-End Benchmark
    # --------------------------------------------------------

    (
        patch_e2e,
        patch_e2e_rows,
    ) = benchmark_end_to_end(
        name=
            "patch_layer3",

        test_paths=
            test_paths,

        preprocess=
            preprocess,

        score_function=
            lambda tensor:
                patch_score(
                    patch_model,
                    patch_coreset,
                    tensor,
                    device,
                ),

        device=
            device,
    )

    # ========================================================
    # 5. Summary Rows
    # ========================================================

    summary_rows = [
        {
            "method":
                "global",

            "startup_ms":
                global_startup_ms,

            "offline_preparation_ms":
                global_offline_ms,

            "representation_bytes":
                global_rep_bytes,

            "representation_kib":
                global_rep_bytes
                / 1024.0,

            "shared_preprocess_mean_ms":
                preprocess_summary[
                    "mean_latency_ms"
                ],

            "shared_preprocess_p95_ms":
                preprocess_summary[
                    "p95_latency_ms"
                ],

            "model_mean_latency_ms":
                global_benchmark[
                    "mean_latency_ms"
                ],

            "model_median_latency_ms":
                global_benchmark[
                    "median_latency_ms"
                ],

            "model_p95_latency_ms":
                global_benchmark[
                    "p95_latency_ms"
                ],

            "model_min_latency_ms":
                global_benchmark[
                    "min_latency_ms"
                ],

            "model_max_latency_ms":
                global_benchmark[
                    "max_latency_ms"
                ],

            "model_throughput_images_s":
                global_benchmark[
                    "throughput_images_s"
                ],

            "peak_allocated_vram_mib":
                global_benchmark[
                    "peak_allocated_vram_mib"
                ],

            "e2e_mean_latency_ms":
                global_e2e[
                    "mean_latency_ms"
                ],

            "e2e_median_latency_ms":
                global_e2e[
                    "median_latency_ms"
                ],

            "e2e_p95_latency_ms":
                global_e2e[
                    "p95_latency_ms"
                ],

            "e2e_min_latency_ms":
                global_e2e[
                    "min_latency_ms"
                ],

            "e2e_max_latency_ms":
                global_e2e[
                    "max_latency_ms"
                ],

            "e2e_throughput_images_s":
                global_e2e[
                    "throughput_images_s"
                ],

            "auroc":
                global_accuracy[
                    "auroc"
                ],

            "ap":
                global_accuracy[
                    "ap"
                ],

            "expected_auroc":
                global_expected_auroc,

            "auroc_delta":
                global_delta,
        },

        {
            "method":
                "patch_layer3",

            "startup_ms":
                patch_startup_ms,

            "offline_preparation_ms":
                patch_offline_ms,

            "representation_bytes":
                patch_rep_bytes,

            "representation_kib":
                patch_rep_bytes
                / 1024.0,

            "shared_preprocess_mean_ms":
                preprocess_summary[
                    "mean_latency_ms"
                ],

            "shared_preprocess_p95_ms":
                preprocess_summary[
                    "p95_latency_ms"
                ],

            "model_mean_latency_ms":
                patch_benchmark[
                    "mean_latency_ms"
                ],

            "model_median_latency_ms":
                patch_benchmark[
                    "median_latency_ms"
                ],

            "model_p95_latency_ms":
                patch_benchmark[
                    "p95_latency_ms"
                ],

            "model_min_latency_ms":
                patch_benchmark[
                    "min_latency_ms"
                ],

            "model_max_latency_ms":
                patch_benchmark[
                    "max_latency_ms"
                ],

            "model_throughput_images_s":
                patch_benchmark[
                    "throughput_images_s"
                ],

            "peak_allocated_vram_mib":
                patch_benchmark[
                    "peak_allocated_vram_mib"
                ],

            "e2e_mean_latency_ms":
                patch_e2e[
                    "mean_latency_ms"
                ],

            "e2e_median_latency_ms":
                patch_e2e[
                    "median_latency_ms"
                ],

            "e2e_p95_latency_ms":
                patch_e2e[
                    "p95_latency_ms"
                ],

            "e2e_min_latency_ms":
                patch_e2e[
                    "min_latency_ms"
                ],

            "e2e_max_latency_ms":
                patch_e2e[
                    "max_latency_ms"
                ],

            "e2e_throughput_images_s":
                patch_e2e[
                    "throughput_images_s"
                ],

            "auroc":
                patch_accuracy[
                    "auroc"
                ],

            "ap":
                patch_accuracy[
                    "ap"
                ],

            "expected_auroc":
                patch_expected_auroc,

            "auroc_delta":
                patch_delta,
        },
    ]

    # ========================================================
    # 6. Save Results
    # ========================================================

    output_dir = Path(
        "results/exp022"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Summary CSV
    # --------------------------------------------------------

    summary_csv = (
        output_dir
        / "engineering_benchmark_summary.csv"
    )

    with open(
        summary_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = (
            csv.DictWriter(
                f,
                fieldnames=
                    list(
                        summary_rows[0]
                        .keys()
                    ),
            )
        )

        writer.writeheader()

        writer.writerows(
            summary_rows
        )

    # --------------------------------------------------------
    # Latency Samples CSV
    # --------------------------------------------------------

    latency_csv = (
        output_dir
        / "engineering_latency_samples.csv"
    )

    all_latency_rows = (
        global_latency_rows
        + global_e2e_rows
        + patch_latency_rows
        + patch_e2e_rows
    )

    with open(
        latency_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = (
            csv.DictWriter(
                f,
                fieldnames=
                    list(
                        all_latency_rows[0]
                        .keys()
                    ),
            )
        )

        writer.writeheader()

        writer.writerows(
            all_latency_rows
        )

    # ========================================================
    # 7. Final Comparison
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL SUMMARY"
    )

    print(
        "=" * 70
    )

    for row in summary_rows:

        print(
            f"\n{row['method']}"
        )

        print(
            f"  AUROC: "
            f"{row['auroc']:.6f}"
        )

        print(
            f"  AP: "
            f"{row['ap']:.6f}"
        )

        print(
            "\n  Model-only:"
        )

        print(
            f"    Mean latency: "
            f"{row['model_mean_latency_ms']:.3f} ms"
        )

        print(
            f"    P95 latency: "
            f"{row['model_p95_latency_ms']:.3f} ms"
        )

        print(
            f"    Throughput: "
            f"{row['model_throughput_images_s']:.3f} "
            f"images/s"
        )

        print(
            "\n  End-to-end:"
        )

        print(
            f"    Mean latency: "
            f"{row['e2e_mean_latency_ms']:.3f} ms"
        )

        print(
            f"    P95 latency: "
            f"{row['e2e_p95_latency_ms']:.3f} ms"
        )

        print(
            f"    Throughput: "
            f"{row['e2e_throughput_images_s']:.3f} "
            f"images/s"
        )

        print(
            "\n  Engineering:"
        )

        print(
            f"    Startup: "
            f"{row['startup_ms']:.3f} ms"
        )

        print(
            f"    Offline preparation: "
            f"{row['offline_preparation_ms']:.3f} ms"
        )

        print(
            f"    Representation: "
            f"{row['representation_kib']:.2f} KiB"
        )

        print(
            f"    Peak allocated VRAM: "
            f"{row['peak_allocated_vram_mib']:.2f} MiB"
        )

    print(
        "\nShared preprocessing:"
    )

    print(
        f"  Mean: "
        f"{preprocess_summary['mean_latency_ms']:.3f} ms"
    )

    print(
        f"  P95: "
        f"{preprocess_summary['p95_latency_ms']:.3f} ms"
    )

    print(
        "\nPatch full memory shape:"
    )

    print(
        patch_full_memory_shape
    )

    print(
        "\nMeasurement scopes:"
    )

    print(
        "  model_only:"
    )

    print(
        "    CPU tensor -> GPU "
        "-> backbone -> anomaly score"
    )

    print(
        "  end_to_end:"
    )

    print(
        "    disk PNG -> PIL -> preprocess "
        "-> GPU -> backbone -> anomaly score"
    )

    print(
        "\nSummary CSV:"
    )

    print(
        summary_csv.resolve()
    )

    print(
        "\nLatency CSV:"
    )

    print(
        latency_csv.resolve()
    )

    print(
        "\nEXP022 active modification "
        "completed."
    )


if __name__ == "__main__":

    main()