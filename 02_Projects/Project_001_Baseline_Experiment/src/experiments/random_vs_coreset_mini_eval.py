from pathlib import Path
import csv

import torch
from sklearn.metrics import roc_auc_score
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


def score_patches_exact(
    query_patches: torch.Tensor,
    memory_bank: torch.Tensor,
):
    """
    Exact nearest-neighbor scoring.

    query_patches:
        [784, 128]

    memory_bank:
        [M, 128]

    Returns:
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

    return patch_scores


def summarize_method(
    name: str,
    labels,
    image_scores,
):
    """
    Summarize image-level anomaly scores.

    label:
        0 = good
        1 = broken_small
    """

    labels_tensor = torch.tensor(
        labels,
        dtype=torch.long,
    )

    scores_tensor = torch.tensor(
        image_scores,
        dtype=torch.float32,
    )

    good_scores = scores_tensor[
        labels_tensor == 0
    ]

    defect_scores = scores_tensor[
        labels_tensor == 1
    ]

    auroc = roc_auc_score(
        labels,
        image_scores,
    )

    good_mean = (
        good_scores.mean().item()
    )

    good_min = (
        good_scores.min().item()
    )

    good_max = (
        good_scores.max().item()
    )

    good_p95 = torch.quantile(
        good_scores,
        0.95,
    ).item()

    defect_mean = (
        defect_scores.mean().item()
    )

    defect_min = (
        defect_scores.min().item()
    )

    defect_max = (
        defect_scores.max().item()
    )

    # Positive margin means:
    #
    # every defect score
    # is above every good score.
    #
    # Negative margin means:
    # score distributions overlap.

    separation_margin = (
        defect_min
        - good_max
    )

    print("\n" + "=" * 70)
    print(f"{name}")
    print("=" * 70)

    print("\nGOOD scores:")

    print(
        f"  min  = {good_min:.6f}"
    )

    print(
        f"  mean = {good_mean:.6f}"
    )

    print(
        f"  P95  = {good_p95:.6f}"
    )

    print(
        f"  max  = {good_max:.6f}"
    )

    print("\nBROKEN_SMALL scores:")

    print(
        f"  min  = {defect_min:.6f}"
    )

    print(
        f"  mean = {defect_mean:.6f}"
    )

    print(
        f"  max  = {defect_max:.6f}"
    )

    print(
        f"\nSeparation margin "
        f"(defect_min - good_max): "
        f"{separation_margin:.6f}"
    )

    print(
        f"\nImage-level AUROC: "
        f"{auroc:.6f}"
    )

    return {
        "good_min": good_min,
        "good_mean": good_mean,
        "good_p95": good_p95,
        "good_max": good_max,

        "defect_min": defect_min,
        "defect_mean": defect_mean,
        "defect_max": defect_max,

        "separation_margin":
            separation_margin,

        "auroc":
            float(auroc),
    }


def main():

    print("=" * 70)
    print("EXP015 - Random vs Coreset Mini Evaluation")
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

    full_memory_bank = build_memory_bank(
        model=model,
        preprocess=preprocess,
        train_good_dir=train_good_dir,
        device=device,
    )

    print("\nFull memory bank:")
    print(full_memory_bank.shape)

    # ========================================================
    # 4. Load EXP014 Selection Result
    # ========================================================

    exp014_path = Path(
        "results/exp014/"
        "minimal_coreset_experiment.pt"
    )

    if not exp014_path.exists():
        raise FileNotFoundError(
            f"EXP014 result not found:\n"
            f"{exp014_path.resolve()}"
        )

    exp014 = torch.load(
        exp014_path,
        map_location="cpu",
    )

    candidate_indices = exp014[
        "candidate_indices"
    ]

    random_indices = exp014[
        "random_indices"
    ]

    coreset_indices = exp014[
        "coreset_indices"
    ]

    # ========================================================
    # 5. Reconstruct EXACT EXP014 Banks
    # ========================================================

    candidate_bank = full_memory_bank[
        candidate_indices
    ]

    random_bank = candidate_bank[
        random_indices
    ]

    coreset_bank = candidate_bank[
        coreset_indices
    ]

    print("\nMemory banks:")

    print(
        f"Candidate reference: "
        f"{tuple(candidate_bank.shape)}"
    )

    print(
        f"Random 100: "
        f"{tuple(random_bank.shape)}"
    )

    print(
        f"Coreset 100: "
        f"{tuple(coreset_bank.shape)}"
    )

    # ========================================================
    # 6. Evaluation Images
    # ========================================================

    good_dir = Path(
        "data/raw/mvtec_anomaly_detection/"
        "bottle/test/good"
    )

    defect_dir = Path(
        "data/raw/mvtec_anomaly_detection/"
        "bottle/test/broken_small"
    )

    num_per_class = 20

    good_paths = sorted(
        good_dir.glob("*.png")
    )[:num_per_class]

    defect_paths = sorted(
        defect_dir.glob("*.png")
    )[:num_per_class]

    if len(good_paths) != num_per_class:
        raise RuntimeError(
            f"Expected {num_per_class} good images, "
            f"found {len(good_paths)}"
        )

    if len(defect_paths) != num_per_class:
        raise RuntimeError(
            f"Expected {num_per_class} broken_small images, "
            f"found {len(defect_paths)}"
        )

    image_paths = (
        good_paths
        + defect_paths
    )

    labels = (
        [0] * len(good_paths)
        + [1] * len(defect_paths)
    )

    print(
        f"\nEvaluation images: "
        f"{len(image_paths)}"
    )

    print(
        f"Good: "
        f"{len(good_paths)}"
    )

    print(
        f"Broken small: "
        f"{len(defect_paths)}"
    )

    # ========================================================
    # 7. Results
    # ========================================================

    candidate_scores = []
    random_scores = []
    coreset_scores = []

    csv_rows = []

    # ========================================================
    # 8. Evaluate
    #
    # Important:
    # Extract image features ONCE,
    # then evaluate all three memory banks.
    # ========================================================

    for index, (
        image_path,
        label,
    ) in enumerate(
        zip(
            image_paths,
            labels,
        ),
        start=1,
    ):

        class_name = (
            "good"
            if label == 0
            else "broken_small"
        )

        print(
            f"\n[{index:02d}/"
            f"{len(image_paths):02d}] "
            f"{class_name}/"
            f"{image_path.name}"
        )

        (
            query_patches,
            feature_height,
            feature_width,
        ) = extract_image_patches(
            image_path=image_path,
            model=model,
            preprocess=preprocess,
            device=device,
        )

        # ----------------------------------------------------
        # Candidate 5000
        # ----------------------------------------------------

        candidate_patch_scores = (
            score_patches_exact(
                query_patches,
                candidate_bank,
            )
        )

        candidate_image_score = (
            candidate_patch_scores.max().item()
        )

        # ----------------------------------------------------
        # Random 100
        # ----------------------------------------------------

        random_patch_scores = (
            score_patches_exact(
                query_patches,
                random_bank,
            )
        )

        random_image_score = (
            random_patch_scores.max().item()
        )

        # ----------------------------------------------------
        # Coreset 100
        # ----------------------------------------------------

        coreset_patch_scores = (
            score_patches_exact(
                query_patches,
                coreset_bank,
            )
        )

        coreset_image_score = (
            coreset_patch_scores.max().item()
        )

        candidate_scores.append(
            candidate_image_score
        )

        random_scores.append(
            random_image_score
        )

        coreset_scores.append(
            coreset_image_score
        )

        print(
            f"  Candidate5000 = "
            f"{candidate_image_score:.6f}"
        )

        print(
            f"  Random100     = "
            f"{random_image_score:.6f}"
        )

        print(
            f"  Coreset100    = "
            f"{coreset_image_score:.6f}"
        )

        csv_rows.append(
            {
                "image": str(
                    image_path
                ),
                "class": class_name,
                "label": label,

                "candidate5000":
                    candidate_image_score,

                "random100":
                    random_image_score,

                "coreset100":
                    coreset_image_score,
            }
        )

    # ========================================================
    # 9. Summary
    # ========================================================

    candidate_summary = summarize_method(
        name="Candidate 5000",
        labels=labels,
        image_scores=candidate_scores,
    )

    random_summary = summarize_method(
        name="Random 100",
        labels=labels,
        image_scores=random_scores,
    )

    coreset_summary = summarize_method(
        name="K-center Coreset 100",
        labels=labels,
        image_scores=coreset_scores,
    )

    # ========================================================
    # 10. Direct Random vs Coreset Comparison
    # ========================================================

    print("\n" + "=" * 70)
    print("Random 100 vs Coreset 100")
    print("=" * 70)

    print(
        "\nGood P95:"
    )

    print(
        f"  Random  = "
        f"{random_summary['good_p95']:.6f}"
    )

    print(
        f"  Coreset = "
        f"{coreset_summary['good_p95']:.6f}"
    )

    print(
        "\nGood Max:"
    )

    print(
        f"  Random  = "
        f"{random_summary['good_max']:.6f}"
    )

    print(
        f"  Coreset = "
        f"{coreset_summary['good_max']:.6f}"
    )

    print(
        "\nSeparation margin:"
    )

    print(
        f"  Random  = "
        f"{random_summary['separation_margin']:.6f}"
    )

    print(
        f"  Coreset = "
        f"{coreset_summary['separation_margin']:.6f}"
    )

    print(
        "\nAUROC:"
    )

    print(
        f"  Random  = "
        f"{random_summary['auroc']:.6f}"
    )

    print(
        f"  Coreset = "
        f"{coreset_summary['auroc']:.6f}"
    )

    # ========================================================
    # 11. Save CSV
    # ========================================================

    output_dir = Path(
        "results/exp015"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "mini_eval_scores.csv"
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
                "image",
                "class",
                "label",
                "candidate5000",
                "random100",
                "coreset100",
            ],
        )

        writer.writeheader()

        writer.writerows(
            csv_rows
        )

    # ========================================================
    # 12. Save Summary
    # ========================================================

    summary_path = (
        output_dir
        / "mini_eval_summary.pt"
    )

    torch.save(
        {
            "candidate_summary":
                candidate_summary,

            "random_summary":
                random_summary,

            "coreset_summary":
                coreset_summary,

            "candidate_scores":
                torch.tensor(
                    candidate_scores
                ),

            "random_scores":
                torch.tensor(
                    random_scores
                ),

            "coreset_scores":
                torch.tensor(
                    coreset_scores
                ),

            "labels":
                torch.tensor(
                    labels
                ),
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