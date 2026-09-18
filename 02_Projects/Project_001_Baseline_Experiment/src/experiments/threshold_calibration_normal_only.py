from pathlib import Path

from PIL import Image

import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
)

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
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


# ============================================================
# Dataset for Global Feature
# ============================================================

class ImageDataset(Dataset):

    def __init__(
        self,
        image_paths,
        preprocess,
    ):
        self.image_paths = list(
            image_paths
        )

        self.preprocess = preprocess

    def __len__(self):
        return len(
            self.image_paths
        )

    def __getitem__(
        self,
        index,
    ):

        image = Image.open(
            self.image_paths[index]
        ).convert("RGB")

        return self.preprocess(
            image
        )


# ============================================================
# Global Feature Extraction
# ============================================================

@torch.inference_mode()
def extract_global_features(
    model,
    image_paths,
    preprocess,
    device,
    batch_size=16,
):

    dataset = ImageDataset(
        image_paths=image_paths,
        preprocess=preprocess,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    feature_parts = []

    for images in loader:

        images = images.to(
            device
        )

        features = model(
            images
        )

        feature_parts.append(
            features.cpu()
        )

    return torch.cat(
        feature_parts,
        dim=0,
    )


# ============================================================
# Patch Memory From Specific Paths
# ============================================================

@torch.inference_mode()
def build_patch_memory_from_paths(
    image_paths,
    model,
    preprocess,
    device,
):

    memory_parts = []

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        (
            patches,
            _,
            _,
        ) = extract_image_patches(
            image_path=image_path,
            model=model,
            preprocess=preprocess,
            device=device,
        )

        memory_parts.append(
            patches.cpu()
        )

        if (
            index % 25 == 0
            or index == len(image_paths)
        ):
            print(
                f"  patch memory: "
                f"{index}/{len(image_paths)}"
            )

    return torch.cat(
        memory_parts,
        dim=0,
    )


# ============================================================
# Build Minimal K-center Coreset
# ============================================================

def build_coreset100(
    full_memory_bank,
    candidate_size=5000,
    selected_size=100,
    seed=42,
):

    generator = torch.Generator()
    generator.manual_seed(
        seed
    )

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

    random_order = torch.randperm(
        candidate_size,
        generator=generator,
    )

    first_index = (
        random_order[0].item()
    )

    coreset_indices = kcenter_greedy(
        candidates=candidate_bank,
        num_selected=selected_size,
        first_index=first_index,
    )

    coreset_bank = candidate_bank[
        coreset_indices
    ]

    return (
        coreset_bank,
        candidate_indices,
        coreset_indices,
    )


# ============================================================
# Patch Image Score
# ============================================================

@torch.inference_mode()
def score_patch_image(
    image_path,
    model,
    preprocess,
    device,
    memory_bank,
):

    (
        query_patches,
        _,
        _,
    ) = extract_image_patches(
        image_path=image_path,
        model=model,
        preprocess=preprocess,
        device=device,
    )

    distances = torch.cdist(
        query_patches,
        memory_bank,
        p=2,
    )

    patch_scores = (
        distances.min(
            dim=1
        ).values
    )

    return (
        patch_scores.max().item()
    )


# ============================================================
# Threshold Metrics
# ============================================================

def confusion_at_threshold(
    labels,
    scores,
    threshold,
):

    labels = torch.tensor(
        labels,
        dtype=torch.long,
    )

    scores = torch.tensor(
        scores,
        dtype=torch.float32,
    )

    predictions = (
        scores >= threshold
    ).long()

    tp = int(
        (
            (predictions == 1)
            &
            (labels == 1)
        ).sum()
    )

    fp = int(
        (
            (predictions == 1)
            &
            (labels == 0)
        ).sum()
    )

    tn = int(
        (
            (predictions == 0)
            &
            (labels == 0)
        ).sum()
    )

    fn = int(
        (
            (predictions == 0)
            &
            (labels == 1)
        ).sum()
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    accuracy = (
        (tp + tn)
        / len(labels)
    )

    f1 = (
        2
        * precision
        * recall
        / (precision + recall)
        if (
            precision + recall
        ) > 0
        else 0.0
    )

    return {
        "threshold":
            float(threshold),

        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,

        "precision":
            precision,

        "recall":
            recall,

        "fpr":
            fpr,

        "specificity":
            specificity,

        "accuracy":
            accuracy,

        "f1":
            f1,
    }


def print_threshold_result(
    name,
    result,
):

    print("\n" + "-" * 70)

    print(name)

    print("-" * 70)

    print(
        f"Threshold   = "
        f"{result['threshold']:.6f}"
    )

    print(
        f"TP / FP     = "
        f"{result['tp']} / "
        f"{result['fp']}"
    )

    print(
        f"TN / FN     = "
        f"{result['tn']} / "
        f"{result['fn']}"
    )

    print(
        f"Precision   = "
        f"{result['precision']:.6f}"
    )

    print(
        f"Recall      = "
        f"{result['recall']:.6f}"
    )

    print(
        f"FPR         = "
        f"{result['fpr']:.6f}"
    )

    print(
        f"Specificity = "
        f"{result['specificity']:.6f}"
    )

    print(
        f"Accuracy    = "
        f"{result['accuracy']:.6f}"
    )

    print(
        f"F1          = "
        f"{result['f1']:.6f}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "EXP018 - Leakage-Safe Threshold Calibration"
    )
    print("=" * 70)

    # ========================================================
    # 1. Reproducibility
    # ========================================================

    split_seed = 2026

    split_generator = (
        torch.Generator()
    )

    split_generator.manual_seed(
        split_seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    print(
        f"Split seed: "
        f"{split_seed}"
    )

    # ========================================================
    # 2. Train-Good Split
    # ========================================================

    train_good_dir = Path(
        "data/raw/mvtec_anomaly_detection/"
        "bottle/train/good"
    )

    all_train_paths = sorted(
        train_good_dir.glob(
            "*.png"
        )
    )

    num_train = len(
        all_train_paths
    )

    permutation = torch.randperm(
        num_train,
        generator=split_generator,
    )

    num_reference = int(
        0.80 * num_train
    )

    reference_indices = (
        permutation[:num_reference]
    )

    validation_indices = (
        permutation[num_reference:]
    )

    reference_paths = [
        all_train_paths[i]
        for i in reference_indices.tolist()
    ]

    validation_paths = [
        all_train_paths[i]
        for i in validation_indices.tolist()
    ]

    print(
        f"\nAll train good: "
        f"{num_train}"
    )

    print(
        f"Reference normal: "
        f"{len(reference_paths)}"
    )

    print(
        f"Validation normal: "
        f"{len(validation_paths)}"
    )

    reference_set = set(
        reference_paths
    )

    validation_set = set(
        validation_paths
    )

    overlap = (
        reference_set
        &
        validation_set
    )

    print(
        f"Reference/validation "
        f"overlap: {len(overlap)}"
    )

    if len(overlap) != 0:
        raise RuntimeError(
            "Reference/validation leakage detected."
        )

    # ========================================================
    # 3. Test Set
    # ========================================================

    test_root = Path(
        "data/raw/mvtec_anomaly_detection/"
        "bottle/test"
    )

    class_names = [
        "good",
        "broken_large",
        "broken_small",
        "contamination",
    ]

    test_paths = []
    test_labels = []
    test_classes = []

    for class_name in class_names:

        paths = sorted(
            (
                test_root
                / class_name
            ).glob("*.png")
        )

        for path in paths:

            test_paths.append(
                path
            )

            test_classes.append(
                class_name
            )

            test_labels.append(
                0
                if class_name == "good"
                else 1
            )

    print(
        f"\nTest images: "
        f"{len(test_paths)}"
    )

    # ========================================================
    # 4. Models
    # ========================================================

    weights = (
        ResNet18_Weights.DEFAULT
    )

    preprocess = (
        weights.transforms()
    )

    global_model = resnet18(
        weights=weights
    )

    global_model.fc = (
        torch.nn.Identity()
    )

    global_model.eval()
    global_model.to(
        device
    )

    patch_model = resnet18(
        weights=weights
    )

    patch_model.eval()
    patch_model.to(
        device
    )

    # ========================================================
    # 5. GLOBAL - Reference Model
    # ========================================================

    print(
        "\nBuilding GLOBAL normal reference..."
    )

    global_reference_features = (
        extract_global_features(
            model=global_model,
            image_paths=reference_paths,
            preprocess=preprocess,
            device=device,
        )
    )

    global_center = (
        global_reference_features.mean(
            dim=0
        )
    )

    # ========================================================
    # 6. GLOBAL - Validation Scores
    # ========================================================

    global_val_features = (
        extract_global_features(
            model=global_model,
            image_paths=validation_paths,
            preprocess=preprocess,
            device=device,
        )
    )

    global_val_scores = (
        torch.linalg.vector_norm(
            global_val_features
            - global_center.unsqueeze(0),
            ord=2,
            dim=1,
        )
    )

    global_t95 = torch.quantile(
        global_val_scores,
        0.95,
    ).item()

    global_t99 = torch.quantile(
        global_val_scores,
        0.99,
    ).item()

    print(
        "\nGLOBAL validation scores:"
    )

    print(
        f"  mean = "
        f"{global_val_scores.mean().item():.6f}"
    )

    print(
        f"  max  = "
        f"{global_val_scores.max().item():.6f}"
    )

    print(
        f"  t95  = "
        f"{global_t95:.6f}"
    )

    print(
        f"  t99  = "
        f"{global_t99:.6f}"
    )

    # ========================================================
    # 7. GLOBAL - Test Scores
    # ========================================================

    global_test_features = (
        extract_global_features(
            model=global_model,
            image_paths=test_paths,
            preprocess=preprocess,
            device=device,
        )
    )

    global_test_scores = (
        torch.linalg.vector_norm(
            global_test_features
            - global_center.unsqueeze(0),
            ord=2,
            dim=1,
        )
    )

    # ========================================================
    # 8. PATCH - Reference Memory
    # ========================================================

    print(
        "\nBuilding PATCH reference memory..."
    )

    full_patch_memory = (
        build_patch_memory_from_paths(
            image_paths=reference_paths,
            model=patch_model,
            preprocess=preprocess,
            device=device,
        )
    )

    print(
        "\nReference patch memory:"
    )

    print(
        full_patch_memory.shape
    )

    # ========================================================
    # 9. PATCH - Coreset100
    # ========================================================

    print(
        "\nBuilding Coreset100..."
    )

    (
        patch_coreset,
        candidate_indices,
        coreset_indices,
    ) = build_coreset100(
        full_memory_bank=
            full_patch_memory,

        candidate_size=5000,

        selected_size=100,

        seed=42,
    )

    print(
        "\nPatch Coreset:"
    )

    print(
        patch_coreset.shape
    )

    # ========================================================
    # 10. PATCH - Validation Scores
    # ========================================================

    patch_val_scores = []

    print(
        "\nScoring PATCH validation normal..."
    )

    for index, path in enumerate(
        validation_paths,
        start=1,
    ):

        score = score_patch_image(
            image_path=path,
            model=patch_model,
            preprocess=preprocess,
            device=device,
            memory_bank=patch_coreset,
        )

        patch_val_scores.append(
            score
        )

        if (
            index % 10 == 0
            or index == len(
                validation_paths
            )
        ):
            print(
                f"  validation: "
                f"{index}/"
                f"{len(validation_paths)}"
            )

    patch_val_scores = torch.tensor(
        patch_val_scores,
        dtype=torch.float32,
    )

    patch_t95 = torch.quantile(
        patch_val_scores,
        0.95,
    ).item()

    patch_t99 = torch.quantile(
        patch_val_scores,
        0.99,
    ).item()

    print(
        "\nPATCH validation scores:"
    )

    print(
        f"  mean = "
        f"{patch_val_scores.mean().item():.6f}"
    )

    print(
        f"  max  = "
        f"{patch_val_scores.max().item():.6f}"
    )

    print(
        f"  t95  = "
        f"{patch_t95:.6f}"
    )

    print(
        f"  t99  = "
        f"{patch_t99:.6f}"
    )

    # ========================================================
    # 11. PATCH - Test Scores
    # ========================================================

    patch_test_scores = []

    print(
        "\nScoring PATCH test set..."
    )

    for index, path in enumerate(
        test_paths,
        start=1,
    ):

        score = score_patch_image(
            image_path=path,
            model=patch_model,
            preprocess=preprocess,
            device=device,
            memory_bank=patch_coreset,
        )

        patch_test_scores.append(
            score
        )

        if (
            index % 10 == 0
            or index == len(
                test_paths
            )
        ):
            print(
                f"  test: "
                f"{index}/"
                f"{len(test_paths)}"
            )

    patch_test_scores = torch.tensor(
        patch_test_scores,
        dtype=torch.float32,
    )

    # ========================================================
    # 12. Ranking Metrics
    # ========================================================

    global_auroc = roc_auc_score(
        test_labels,
        global_test_scores.numpy(),
    )

    global_ap = average_precision_score(
        test_labels,
        global_test_scores.numpy(),
    )

    patch_auroc = roc_auc_score(
        test_labels,
        patch_test_scores.numpy(),
    )

    patch_ap = average_precision_score(
        test_labels,
        patch_test_scores.numpy(),
    )

    print("\n" + "=" * 70)
    print("Ranking Metrics")
    print("=" * 70)

    print(
        f"\nGlobal AUROC = "
        f"{global_auroc:.6f}"
    )

    print(
        f"Global AP    = "
        f"{global_ap:.6f}"
    )

    print(
        f"\nPatch AUROC  = "
        f"{patch_auroc:.6f}"
    )

    print(
        f"Patch AP     = "
        f"{patch_ap:.6f}"
    )

    # ========================================================
    # 13. ROC / PR Curves
    #
    # Diagnostic only.
    # We DO NOT choose threshold from test labels.
    # ========================================================

    (
        global_fpr_curve,
        global_tpr_curve,
        global_roc_thresholds,
    ) = roc_curve(
        test_labels,
        global_test_scores.numpy(),
    )

    (
        global_precision_curve,
        global_recall_curve,
        global_pr_thresholds,
    ) = precision_recall_curve(
        test_labels,
        global_test_scores.numpy(),
    )

    (
        patch_fpr_curve,
        patch_tpr_curve,
        patch_roc_thresholds,
    ) = roc_curve(
        test_labels,
        patch_test_scores.numpy(),
    )

    (
        patch_precision_curve,
        patch_recall_curve,
        patch_pr_thresholds,
    ) = precision_recall_curve(
        test_labels,
        patch_test_scores.numpy(),
    )

    # ========================================================
    # 14. Threshold Metrics
    # ========================================================

    global_95 = confusion_at_threshold(
        labels=test_labels,
        scores=global_test_scores.tolist(),
        threshold=global_t95,
    )

    global_99 = confusion_at_threshold(
        labels=test_labels,
        scores=global_test_scores.tolist(),
        threshold=global_t99,
    )

    patch_95 = confusion_at_threshold(
        labels=test_labels,
        scores=patch_test_scores.tolist(),
        threshold=patch_t95,
    )

    patch_99 = confusion_at_threshold(
        labels=test_labels,
        scores=patch_test_scores.tolist(),
        threshold=patch_t99,
    )

    print("\n" + "=" * 70)
    print("GLOBAL Threshold Evaluation")
    print("=" * 70)

    print_threshold_result(
        "GLOBAL - Validation P95",
        global_95,
    )

    print_threshold_result(
        "GLOBAL - Validation P99",
        global_99,
    )

    print("\n" + "=" * 70)
    print("PATCH Threshold Evaluation")
    print("=" * 70)

    print_threshold_result(
        "PATCH - Validation P95",
        patch_95,
    )

    print_threshold_result(
        "PATCH - Validation P99",
        patch_99,
    )

    # ========================================================
    # 15. Monotonicity Checks
    # ========================================================

    print("\n" + "=" * 70)
    print("Threshold Monotonicity Checks")
    print("=" * 70)

    print(
        f"\nGlobal t99 >= t95: "
        f"{global_t99 >= global_t95}"
    )

    print(
        f"Global FPR99 <= FPR95: "
        f"{global_99['fpr'] <= global_95['fpr']}"
    )

    print(
        f"Global Recall99 <= Recall95: "
        f"{global_99['recall'] <= global_95['recall']}"
    )

    print(
        f"\nPatch t99 >= t95: "
        f"{patch_t99 >= patch_t95}"
    )

    print(
        f"Patch FPR99 <= FPR95: "
        f"{patch_99['fpr'] <= patch_95['fpr']}"
    )

    print(
        f"Patch Recall99 <= Recall95: "
        f"{patch_99['recall'] <= patch_95['recall']}"
    )

    # ========================================================
    # 16. Save Results
    # ========================================================

    output_dir = Path(
        "results/exp018"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "threshold_calibration.pt"
    )

    torch.save(
        {
            "split_seed":
                split_seed,

            "reference_paths":
                [
                    str(path)
                    for path
                    in reference_paths
                ],

            "validation_paths":
                [
                    str(path)
                    for path
                    in validation_paths
                ],

            "test_paths":
                [
                    str(path)
                    for path
                    in test_paths
                ],

            "test_labels":
                torch.tensor(
                    test_labels
                ),

            "test_classes":
                test_classes,

            "global_val_scores":
                global_val_scores,

            "patch_val_scores":
                patch_val_scores,

            "global_test_scores":
                global_test_scores,

            "patch_test_scores":
                patch_test_scores,

            "global_t95":
                global_t95,

            "global_t99":
                global_t99,

            "patch_t95":
                patch_t95,

            "patch_t99":
                patch_t99,

            "global_95":
                global_95,

            "global_99":
                global_99,

            "patch_95":
                patch_95,

            "patch_99":
                patch_99,

            "global_auroc":
                float(global_auroc),

            "global_ap":
                float(global_ap),

            "patch_auroc":
                float(patch_auroc),

            "patch_ap":
                float(patch_ap),

            "global_roc":
                {
                    "fpr":
                        torch.tensor(
                            global_fpr_curve
                        ),

                    "tpr":
                        torch.tensor(
                            global_tpr_curve
                        ),

                    "thresholds":
                        torch.tensor(
                            global_roc_thresholds
                        ),
                },

            "patch_roc":
                {
                    "fpr":
                        torch.tensor(
                            patch_fpr_curve
                        ),

                    "tpr":
                        torch.tensor(
                            patch_tpr_curve
                        ),

                    "thresholds":
                        torch.tensor(
                            patch_roc_thresholds
                        ),
                },

            "candidate_indices":
                candidate_indices,

            "coreset_indices":
                coreset_indices,
        },
        output_path,
    )

    print("\nResult saved to:")
    print(
        output_path.resolve()
    )

    print("\n" + "=" * 70)
    print("Experiment completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()