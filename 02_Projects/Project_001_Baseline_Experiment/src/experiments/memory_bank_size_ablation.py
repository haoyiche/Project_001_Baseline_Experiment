from pathlib import Path
import time

import torch
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)

from src.experiments.single_patch_nearest_neighbor import (
    build_memory_bank,
)

from src.experiments.full_image_patch_scoring import (
    extract_image_patches,
    nearest_neighbor_scores_chunked,
    decode_patch_index,
)


def summarize_scores(
    name: str,
    scores: torch.Tensor,
    feature_width: int,
):
    """
    Print statistics and Top-5 anomaly patches.
    """

    min_score = scores.min().item()
    mean_score = scores.mean().item()
    max_score = scores.max().item()

    print("\n" + "=" * 70)
    print(f"{name} Score Summary")
    print("=" * 70)

    print(
        f"\nMinimum score: "
        f"{min_score:.6f}"
    )

    print(
        f"Mean score: "
        f"{mean_score:.6f}"
    )

    print(
        f"Maximum score: "
        f"{max_score:.6f}"
    )

    top_scores, top_indices = torch.topk(
        scores,
        k=5,
    )

    print("\nTop-5:")

    top_positions = []

    for rank in range(5):

        patch_index = (
            top_indices[rank].item()
        )

        score = (
            top_scores[rank].item()
        )

        row, col = decode_patch_index(
            patch_index,
            feature_width,
        )

        top_positions.append(
            (row, col)
        )

        print(
            f"\nRank {rank + 1}:"
            f"\n  patch index = {patch_index}"
            f"\n  position    = ({row}, {col})"
            f"\n  score       = {score:.6f}"
        )

    return {
        "min": min_score,
        "mean": mean_score,
        "max": max_score,
        "top_positions": top_positions,
    }


def compare_with_full(
    name: str,
    full_scores: torch.Tensor,
    reduced_scores: torch.Tensor,
):
    """
    Compare reduced-bank scores with the
    full-memory-bank reference.
    """

    difference = (
        reduced_scores
        - full_scores
    )

    mae = torch.mean(
        torch.abs(difference)
    ).item()

    max_abs_difference = torch.max(
        torch.abs(difference)
    ).item()

    # Correlation between spatial score patterns
    correlation_matrix = torch.corrcoef(
        torch.stack(
            [
                full_scores,
                reduced_scores,
            ]
        )
    )

    correlation = (
        correlation_matrix[0, 1].item()
    )

    # Reduced memory bank is a subset of full bank.
    # Therefore reduced NN score should NEVER
    # be smaller than full NN score, except for
    # tiny floating-point differences.

    tolerance = 1e-5

    violations = (
        reduced_scores
        < full_scores - tolerance
    ).sum().item()

    print("\n" + "=" * 70)
    print(f"{name} vs Full Memory Bank")
    print("=" * 70)

    print(
        f"\nMean absolute score difference: "
        f"{mae:.6f}"
    )

    print(
        f"Maximum absolute score difference: "
        f"{max_abs_difference:.6f}"
    )

    print(
        f"Score-map correlation: "
        f"{correlation:.6f}"
    )

    print(
        f"Monotonicity violations "
        f"(reduced < full): "
        f"{violations}"
    )

    return {
        "mae": mae,
        "max_abs_difference": max_abs_difference,
        "correlation": correlation,
        "violations": violations,
    }


def main():

    print("=" * 70)
    print("EXP013 - Memory Bank Size Ablation")
    print("=" * 70)

    # ========================================================
    # 1. Reproducibility
    # ========================================================

    random_seed = 42

    generator = torch.Generator()
    generator.manual_seed(
        random_seed
    )

    print(
        f"\nRandom seed: "
        f"{random_seed}"
    )

    # ========================================================
    # 2. Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: "
        f"{device}"
    )

    # ========================================================
    # 3. Model
    # ========================================================

    weights = ResNet18_Weights.DEFAULT

    preprocess = weights.transforms()

    model = resnet18(
        weights=weights
    )

    model.eval()
    model.to(device)

    # ========================================================
    # 4. Paths
    # ========================================================

    train_good_dir = Path(
        "data/raw/mvtec_anomaly_detection/"
        "bottle/train/good"
    )

    test_image_path = Path(
        "data/raw/mvtec_anomaly_detection/"
        "bottle/test/broken_small/000.png"
    )

    # EXP012 already calculated the exact full-bank scores.
    # Reuse them instead of wasting time calculating
    # the same 128 million pair distances again.

    full_score_path = Path(
        "results/exp012/"
        "broken_small_000_scores.pt"
    )

    if not full_score_path.exists():
        raise FileNotFoundError(
            f"\nEXP012 score file not found:\n"
            f"{full_score_path.resolve()}\n"
            f"Run EXP012 first."
        )

    # ========================================================
    # 5. Load Full-bank Reference Scores
    # ========================================================

    saved_data = torch.load(
        full_score_path,
        map_location="cpu",
    )

    full_scores = (
        saved_data["patch_scores"]
        .cpu()
        .float()
    )

    print("\nFull-bank reference scores:")
    print(full_scores.shape)

    # ========================================================
    # 6. Build Full Normal Memory Bank
    # ========================================================

    print(
        "\nBuilding full normal memory bank..."
    )

    memory_bank = build_memory_bank(
        model=model,
        preprocess=preprocess,
        train_good_dir=train_good_dir,
        device=device,
    )

    print("\nFull memory bank:")
    print(memory_bank.shape)

    num_memory = memory_bank.shape[0]

    # ========================================================
    # 7. Create NESTED Random Subsets
    # ========================================================

    permutation = torch.randperm(
        num_memory,
        generator=generator,
    )

    size_10 = max(
        1,
        int(num_memory * 0.10),
    )

    size_1 = max(
        1,
        int(num_memory * 0.01),
    )

    # Important:
    #
    # 1% is a subset of 10%
    # and 10% is a subset of full bank.

    indices_10 = permutation[
        :size_10
    ]

    indices_1 = permutation[
        :size_1
    ]

    memory_bank_10 = memory_bank[
        indices_10
    ]

    memory_bank_1 = memory_bank[
        indices_1
    ]

    print("\nMemory Bank Shapes")

    print(
        f"\nFull:"
        f"\n{tuple(memory_bank.shape)}"
    )

    print(
        f"\n10%:"
        f"\n{tuple(memory_bank_10.shape)}"
    )

    print(
        f"\n1%:"
        f"\n{tuple(memory_bank_1.shape)}"
    )

    # ========================================================
    # 8. Extract Query Image Once
    # ========================================================

    (
        query_patches,
        feature_height,
        feature_width,
    ) = extract_image_patches(
        image_path=test_image_path,
        model=model,
        preprocess=preprocess,
        device=device,
    )

    print("\nQuery patches:")
    print(query_patches.shape)

    # ========================================================
    # 9. Full-bank Summary
    #
    # No recalculation.
    # Use EXP012 exact scores.
    # ========================================================

    full_summary = summarize_scores(
        name="100% Full Memory",
        scores=full_scores,
        feature_width=feature_width,
    )

    # ========================================================
    # 10. 10% Memory Bank
    # ========================================================

    print("\n" + "#" * 70)
    print("10% MEMORY BANK")
    print("#" * 70)

    start_time = time.perf_counter()

    (
        scores_10,
        nearest_indices_10,
    ) = nearest_neighbor_scores_chunked(
        query_patches=query_patches,
        memory_bank=memory_bank_10,
        memory_chunk_size=4096,
    )

    elapsed_10 = (
        time.perf_counter()
        - start_time
    )

    summary_10 = summarize_scores(
        name="10% Memory",
        scores=scores_10,
        feature_width=feature_width,
    )

    comparison_10 = compare_with_full(
        name="10% Memory",
        full_scores=full_scores,
        reduced_scores=scores_10,
    )

    print(
        f"\n10% NN search time: "
        f"{elapsed_10:.3f} seconds"
    )

    # ========================================================
    # 11. 1% Memory Bank
    # ========================================================

    print("\n" + "#" * 70)
    print("1% MEMORY BANK")
    print("#" * 70)

    start_time = time.perf_counter()

    (
        scores_1,
        nearest_indices_1,
    ) = nearest_neighbor_scores_chunked(
        query_patches=query_patches,
        memory_bank=memory_bank_1,
        memory_chunk_size=4096,
    )

    elapsed_1 = (
        time.perf_counter()
        - start_time
    )

    summary_1 = summarize_scores(
        name="1% Memory",
        scores=scores_1,
        feature_width=feature_width,
    )

    comparison_1 = compare_with_full(
        name="1% Memory",
        full_scores=full_scores,
        reduced_scores=scores_1,
    )

    print(
        f"\n1% NN search time: "
        f"{elapsed_1:.3f} seconds"
    )

    # ========================================================
    # 12. Strong Mathematical Verification
    #
    # Because:
    #
    # M_1% subset M_10% subset M_full
    #
    # Must have:
    #
    # full_score <= 10%_score <= 1%_score
    # ========================================================

    tolerance = 1e-5

    full_le_10 = torch.all(
        full_scores
        <= scores_10 + tolerance
    ).item()

    ten_le_1 = torch.all(
        scores_10
        <= scores_1 + tolerance
    ).item()

    print("\n" + "=" * 70)
    print("Nested Memory Bank Verification")
    print("=" * 70)

    print(
        "\nExpected:"
        "\nscore_full <= score_10% <= score_1%"
    )

    print(
        f"\nFull <= 10% for every patch: "
        f"{full_le_10}"
    )

    print(
        f"10% <= 1% for every patch: "
        f"{ten_le_1}"
    )

    if not full_le_10:
        raise RuntimeError(
            "Unexpected result: "
            "10% subset produced score lower "
            "than full memory bank."
        )

    if not ten_le_1:
        raise RuntimeError(
            "Unexpected result: "
            "1% subset produced score lower "
            "than 10% memory bank."
        )

    # ========================================================
    # 13. Compare Top-5 Spatial Positions
    # ========================================================

    full_top = set(
        full_summary["top_positions"]
    )

    top_10 = set(
        summary_10["top_positions"]
    )

    top_1 = set(
        summary_1["top_positions"]
    )

    overlap_10 = len(
        full_top.intersection(
            top_10
        )
    )

    overlap_1 = len(
        full_top.intersection(
            top_1
        )
    )

    print("\n" + "=" * 70)
    print("Top-5 Spatial Stability")
    print("=" * 70)

    print(
        f"\nFull vs 10% "
        f"Top-5 overlap: "
        f"{overlap_10}/5"
    )

    print(
        f"Full vs 1% "
        f"Top-5 overlap: "
        f"{overlap_1}/5"
    )

    # ========================================================
    # 14. Workload
    # ========================================================

    num_queries = query_patches.shape[0]

    pairs_full = (
        num_queries
        * memory_bank.shape[0]
    )

    pairs_10 = (
        num_queries
        * memory_bank_10.shape[0]
    )

    pairs_1 = (
        num_queries
        * memory_bank_1.shape[0]
    )

    print("\n" + "=" * 70)
    print("Pairwise Distance Workload")
    print("=" * 70)

    print(
        f"\nFull:"
        f"\n{pairs_full:,} patch pairs"
    )

    print(
        f"\n10%:"
        f"\n{pairs_10:,} patch pairs"
    )

    print(
        f"\n1%:"
        f"\n{pairs_1:,} patch pairs"
    )

    # ========================================================
    # 15. Save Small Result File
    # ========================================================

    output_dir = Path(
        "results/exp013"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "memory_bank_size_ablation.pt"
    )

    torch.save(
        {
            "random_seed": random_seed,

            "full_scores": full_scores,
            "scores_10": scores_10,
            "scores_1": scores_1,

            "memory_size_full": (
                memory_bank.shape[0]
            ),

            "memory_size_10": (
                memory_bank_10.shape[0]
            ),

            "memory_size_1": (
                memory_bank_1.shape[0]
            ),

            "elapsed_10": elapsed_10,
            "elapsed_1": elapsed_1,

            "comparison_10": comparison_10,
            "comparison_1": comparison_1,
        },
        output_path,
    )

    print("\nResult saved to:")
    print(output_path.resolve())

    print("\n" + "=" * 70)
    print("Experiment completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()