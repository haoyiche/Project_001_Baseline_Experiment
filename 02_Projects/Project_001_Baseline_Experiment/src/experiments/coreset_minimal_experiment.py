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


def calculate_coverage(
    candidates: torch.Tensor,
    selected: torch.Tensor,
    chunk_size: int = 1000,
):
    """
    For every candidate patch, calculate distance
    to its nearest selected representative.

    candidates:
        [N, C]

    selected:
        [K, C]

    Returns:
        nearest_distances:
            [N]
    """

    nearest_parts = []

    for start in range(
        0,
        candidates.shape[0],
        chunk_size,
    ):

        end = min(
            start + chunk_size,
            candidates.shape[0],
        )

        chunk = candidates[start:end]

        # [chunk, C]
        # vs
        # [K, C]
        #
        # ->
        # [chunk, K]

        distances = torch.cdist(
            chunk,
            selected,
            p=2,
        )

        nearest_distances = (
            distances.min(dim=1).values
        )

        nearest_parts.append(
            nearest_distances
        )

    return torch.cat(
        nearest_parts,
        dim=0,
    )


def kcenter_greedy(
    candidates: torch.Tensor,
    num_selected: int,
    first_index: int,
):
    """
    Minimal exact K-center greedy.

    candidates:
        [N, C]

    Returns:
        selected_indices:
            [K]
    """

    num_candidates = (
        candidates.shape[0]
    )

    if num_selected > num_candidates:
        raise ValueError(
            "num_selected cannot exceed "
            "num_candidates"
        )

    # --------------------------------------------------------
    # 1. Start from one fixed patch
    # --------------------------------------------------------

    selected_indices = [
        first_index
    ]

    first_center = candidates[
        first_index
    ].unsqueeze(0)

    # Distance of every candidate to
    # the current selected set.
    #
    # Initially the selected set contains
    # only one center.

    min_distances = torch.cdist(
        candidates,
        first_center,
        p=2,
    ).squeeze(1)

    # The first selected point itself has
    # distance 0.
    min_distances[
        first_index
    ] = 0.0

    # --------------------------------------------------------
    # 2. Repeatedly choose the farthest patch
    # --------------------------------------------------------

    for step in range(
        1,
        num_selected,
    ):

        # Candidate that is currently worst covered
        next_index = torch.argmax(
            min_distances
        ).item()

        selected_indices.append(
            next_index
        )

        new_center = candidates[
            next_index
        ].unsqueeze(0)

        # Distance to ONLY the new center
        new_distances = torch.cdist(
            candidates,
            new_center,
            p=2,
        ).squeeze(1)

        # Important:
        #
        # Distance to selected set is:
        #
        # min(
        #     previous nearest distance,
        #     distance to new center
        # )

        min_distances = torch.minimum(
            min_distances,
            new_distances,
        )

        # Safety
        min_distances[
            selected_indices
        ] = 0.0

        if (
            step < 10
            or (step + 1) % 10 == 0
            or step == num_selected - 1
        ):
            print(
                f"Selected "
                f"{step + 1:3d}/{num_selected} | "
                f"current coverage radius = "
                f"{min_distances.max().item():.6f}"
            )

    selected_indices = torch.tensor(
        selected_indices,
        dtype=torch.long,
    )

    return selected_indices


def summarize_coverage(
    name: str,
    coverage_distances: torch.Tensor,
):
    """
    Describe how well selected representatives
    cover the candidate normal features.
    """

    minimum = (
        coverage_distances.min().item()
    )

    mean = (
        coverage_distances.mean().item()
    )

    maximum = (
        coverage_distances.max().item()
    )

    p95 = torch.quantile(
        coverage_distances,
        0.95,
    ).item()

    print("\n" + "=" * 70)
    print(f"{name} Coverage")
    print("=" * 70)

    print(
        f"\nMinimum distance: "
        f"{minimum:.6f}"
    )

    print(
        f"Mean distance: "
        f"{mean:.6f}"
    )

    print(
        f"P95 distance: "
        f"{p95:.6f}"
    )

    print(
        f"Maximum distance: "
        f"{maximum:.6f}"
    )

    return {
        "min": minimum,
        "mean": mean,
        "p95": p95,
        "max": maximum,
    }


def main():

    print("=" * 70)
    print("EXP014 - Minimal K-Center Coreset Experiment")
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
    # 4. Build Full Memory Bank
    # ========================================================

    train_good_dir = Path(
        "data/raw/mvtec_anomaly_detection/"
        "bottle/train/good"
    )

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

    # ========================================================
    # 5. Candidate Pool
    # ========================================================

    candidate_size = 5000
    selected_size = 100

    if candidate_size > memory_bank.shape[0]:
        raise ValueError(
            "candidate_size exceeds memory bank"
        )

    permutation = torch.randperm(
        memory_bank.shape[0],
        generator=generator,
    )

    candidate_indices = permutation[
        :candidate_size
    ]

    candidates = memory_bank[
        candidate_indices
    ].clone()

    print("\nCandidate pool:")
    print(candidates.shape)

    print(
        f"\nSelected size: "
        f"{selected_size}"
    )

    print(
        f"Selection ratio: "
        f"{selected_size / candidate_size:.2%}"
    )

    # ========================================================
    # 6. Random Baseline
    # ========================================================

    # Use another deterministic permutation
    # inside the candidate pool.

    random_order = torch.randperm(
        candidate_size,
        generator=generator,
    )

    random_indices = random_order[
        :selected_size
    ]

    random_selected = candidates[
        random_indices
    ]

    print("\nRandom selected:")
    print(random_selected.shape)

    # ========================================================
    # 7. K-center Greedy
    # ========================================================

    # Use the SAME first center as the random
    # baseline's first selected point.
    #
    # This makes the comparison slightly cleaner.

    first_index = (
        random_indices[0].item()
    )

    print(
        f"\nK-center first index: "
        f"{first_index}"
    )

    start_time = time.perf_counter()

    coreset_indices = kcenter_greedy(
        candidates=candidates,
        num_selected=selected_size,
        first_index=first_index,
    )

    coreset_elapsed = (
        time.perf_counter()
        - start_time
    )

    coreset_selected = candidates[
        coreset_indices
    ]

    print("\nCoreset selected:")
    print(coreset_selected.shape)

    print(
        f"\nK-center selection time: "
        f"{coreset_elapsed:.3f} seconds"
    )

    # ========================================================
    # 8. Coverage Evaluation
    # ========================================================

    print(
        "\nEvaluating Random coverage..."
    )

    random_coverage = (
        calculate_coverage(
            candidates=candidates,
            selected=random_selected,
        )
    )

    random_summary = summarize_coverage(
        name="Random 100",
        coverage_distances=random_coverage,
    )

    print(
        "\nEvaluating Coreset coverage..."
    )

    coreset_coverage = (
        calculate_coverage(
            candidates=candidates,
            selected=coreset_selected,
        )
    )

    coreset_summary = summarize_coverage(
        name="K-center Coreset 100",
        coverage_distances=coreset_coverage,
    )

    # ========================================================
    # 9. Comparison
    # ========================================================

    mean_improvement = (
        random_summary["mean"]
        - coreset_summary["mean"]
    )

    p95_improvement = (
        random_summary["p95"]
        - coreset_summary["p95"]
    )

    max_improvement = (
        random_summary["max"]
        - coreset_summary["max"]
    )

    print("\n" + "=" * 70)
    print("Random vs Coreset")
    print("=" * 70)

    print(
        f"\nMean coverage improvement: "
        f"{mean_improvement:.6f}"
    )

    print(
        f"P95 coverage improvement: "
        f"{p95_improvement:.6f}"
    )

    print(
        f"Maximum coverage improvement: "
        f"{max_improvement:.6f}"
    )

    coreset_mean_better = (
        coreset_summary["mean"]
        < random_summary["mean"]
    )

    coreset_p95_better = (
        coreset_summary["p95"]
        < random_summary["p95"]
    )

    coreset_max_better = (
        coreset_summary["max"]
        < random_summary["max"]
    )

    print(
        f"\nCoreset mean better: "
        f"{coreset_mean_better}"
    )

    print(
        f"Coreset P95 better: "
        f"{coreset_p95_better}"
    )

    print(
        f"Coreset max better: "
        f"{coreset_max_better}"
    )

    # ========================================================
    # 10. Save Result
    # ========================================================

    output_dir = Path(
        "results/exp014"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "minimal_coreset_experiment.pt"
    )

    torch.save(
        {
            "random_seed": random_seed,

            "candidate_indices":
                candidate_indices,

            "random_indices":
                random_indices,

            "coreset_indices":
                coreset_indices,

            "candidate_size":
                candidate_size,

            "selected_size":
                selected_size,

            "random_summary":
                random_summary,

            "coreset_summary":
                coreset_summary,

            "coreset_elapsed":
                coreset_elapsed,
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