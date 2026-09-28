import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import (
    ViT_B_16_Weights,
    vit_b_16,
)


# ============================================================
# Config
# ============================================================

MANIFEST_PATH = Path(
    "configs/exp023_coco_subset500_ids.json"
)

ANNOTATION_PATH = Path(
    "data/raw/coco2017/annotations/instances_val2017.json"
)

IMAGE_DIR = Path(
    "data/raw/coco2017/val2017_subset500"
)

OUTPUT_DIR = Path(
    "results/exp024"
)

PER_QUERY_PATH = (
    OUTPUT_DIR
    / "vit_cls_vs_mean_retrieval_per_query.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "vit_cls_vs_mean_retrieval_summary.csv"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

NUM_GALLERY = 400
NUM_QUERY = 100
K = 5
BATCH_SIZE = 8


# ============================================================
# Helpers
# ============================================================

def image_path_from_id(image_id):

    return (
        IMAGE_DIR
        / f"{image_id:012d}.jpg"
    )


def mean(values):

    if not values:
        return float("nan")

    return sum(values) / len(values)


def pair_relevance(
    query_categories,
    gallery_categories,
):

    intersection = (
        query_categories
        & gallery_categories
    )

    union = (
        query_categories
        | gallery_categories
    )

    num_intersection = len(
        intersection
    )

    binary = int(
        num_intersection > 0
    )

    coverage = (
        num_intersection
        / len(query_categories)
        if query_categories
        else float("nan")
    )

    jaccard = (
        num_intersection
        / len(union)
        if union
        else float("nan")
    )

    strict = int(
        bool(query_categories)
        and query_categories.issubset(
            gallery_categories
        )
    )

    return {
        "binary": binary,
        "coverage": coverage,
        "jaccard": jaccard,
        "strict": strict,
    }


def evaluate_topk(
    retrieved_ids,
    query_categories,
    image_categories,
):

    metrics = []

    for gallery_id in retrieved_ids:

        metrics.append(
            pair_relevance(
                query_categories,
                image_categories[
                    gallery_id
                ],
            )
        )

    return {
        "binary_p5": mean(
            [
                x["binary"]
                for x in metrics
            ]
        ),

        "coverage_at5": mean(
            [
                x["coverage"]
                for x in metrics
            ]
        ),

        "jaccard_at5": mean(
            [
                x["jaccard"]
                for x in metrics
            ]
        ),

        "strict_hit5": int(
            any(
                x["strict"]
                for x in metrics
            )
        ),
    }


def compare_pair(
    first,
    second,
    eps=1e-12,
):

    delta = first - second

    if delta > eps:
        return "win"

    if delta < -eps:
        return "loss"

    return "tie"


# ============================================================
# ViT feature extraction
# ============================================================

def extract_vit_embeddings(
    image_ids,
    model,
    transform,
):

    cls_all = []
    mean_patch_all = []

    for start in range(
        0,
        len(image_ids),
        BATCH_SIZE,
    ):

        batch_ids = image_ids[
            start:
            start + BATCH_SIZE
        ]

        tensors = []

        for image_id in batch_ids:

            image = Image.open(
                image_path_from_id(
                    image_id
                )
            ).convert("RGB")

            tensors.append(
                transform(image)
            )

        x = torch.stack(
            tensors,
            dim=0,
        ).to(
            DEVICE
        )

        with torch.inference_mode():

            # --------------------------------------------
            # [B, 3, 224, 224]
            # ->
            # [B, 196, 768]
            # --------------------------------------------

            patch_tokens = (
                model._process_input(
                    x
                )
            )

            batch_size = (
                patch_tokens.shape[0]
            )

            # --------------------------------------------
            # Add CLS
            # [B, 196, 768]
            # ->
            # [B, 197, 768]
            # --------------------------------------------

            cls_token = (
                model.class_token.expand(
                    batch_size,
                    -1,
                    -1,
                )
            )

            tokens = torch.cat(
                [
                    cls_token,
                    patch_tokens,
                ],
                dim=1,
            )

            # --------------------------------------------
            # Transformer encoder
            # --------------------------------------------

            encoded = (
                model.encoder(
                    tokens
                )
            )

            # --------------------------------------------
            # Readout A: CLS
            # [B, 768]
            # --------------------------------------------

            cls_embedding = (
                encoded[:, 0, :]
            )

            # --------------------------------------------
            # Readout B: mean encoded patch tokens
            # [B, 196, 768]
            # ->
            # [B, 768]
            # --------------------------------------------

            patch_embeddings = (
                encoded[:, 1:, :]
            )

            mean_patch_embedding = (
                patch_embeddings.mean(
                    dim=1
                )
            )

        cls_all.append(
            cls_embedding.cpu()
        )

        mean_patch_all.append(
            mean_patch_embedding.cpu()
        )

        print(
            f"Extracted "
            f"{min(start + BATCH_SIZE, len(image_ids))}"
            f"/{len(image_ids)}"
        )

    cls_all = torch.cat(
        cls_all,
        dim=0,
    )

    mean_patch_all = torch.cat(
        mean_patch_all,
        dim=0,
    )

    return (
        cls_all,
        mean_patch_all,
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Fixed split
    # --------------------------------------------------------

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        manifest = json.load(f)

    image_ids = manifest[
        "image_ids"
    ]

    gallery_ids = image_ids[
        :NUM_GALLERY
    ]

    query_ids = image_ids[
        NUM_GALLERY:
        NUM_GALLERY + NUM_QUERY
    ]

    print(
        "Gallery:",
        len(gallery_ids)
    )

    print(
        "Query:",
        len(query_ids)
    )

    print(
        "Device:",
        DEVICE
    )


    # --------------------------------------------------------
    # 2. COCO annotations
    # --------------------------------------------------------

    with open(
        ANNOTATION_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        coco = json.load(f)

    category_names = {
        category["id"]:
        category["name"]
        for category
        in coco["categories"]
    }

    image_categories = defaultdict(
        set
    )

    fixed_ids = set(
        image_ids
    )

    for ann in coco[
        "annotations"
    ]:

        image_id = ann[
            "image_id"
        ]

        if image_id not in fixed_ids:
            continue

        image_categories[
            image_id
        ].add(
            ann["category_id"]
        )


    # --------------------------------------------------------
    # 3. ViT backbone
    # --------------------------------------------------------

    weights = (
        ViT_B_16_Weights.DEFAULT
    )

    transform = (
        weights.transforms()
    )

    model = vit_b_16(
        weights=weights
    )

    model.eval()
    model.to(
        DEVICE
    )


    # --------------------------------------------------------
    # 4. Extract BOTH readouts from SAME forward tokens
    # --------------------------------------------------------

    (
        cls_embeddings,
        mean_patch_embeddings,
    ) = extract_vit_embeddings(
        image_ids,
        model,
        transform,
    )

    print()
    print(
        "CLS embeddings:",
        cls_embeddings.shape,
    )

    print(
        "Mean-patch embeddings:",
        mean_patch_embeddings.shape,
    )


    # --------------------------------------------------------
    # 5. Dataset-level CLS vs mean-patch geometry
    # --------------------------------------------------------

    cls_mean_cosine = (
        F.cosine_similarity(
            cls_embeddings,
            mean_patch_embeddings,
            dim=1,
        )
    )

    print()
    print(
        "=" * 70
    )

    print(
        "CLS VS MEAN-PATCH GEOMETRY"
    )

    print(
        "=" * 70
    )

    print(
        "Cosine min:",
        cls_mean_cosine.min().item(),
    )

    print(
        "Cosine mean:",
        cls_mean_cosine.mean().item(),
    )

    print(
        "Cosine max:",
        cls_mean_cosine.max().item(),
    )


    # --------------------------------------------------------
    # 6. Split gallery / query
    # --------------------------------------------------------

    cls_gallery = (
        cls_embeddings[
            :NUM_GALLERY
        ]
    )

    cls_query = (
        cls_embeddings[
            NUM_GALLERY:
        ]
    )

    mean_gallery = (
        mean_patch_embeddings[
            :NUM_GALLERY
        ]
    )

    mean_query = (
        mean_patch_embeddings[
            NUM_GALLERY:
        ]
    )


    # --------------------------------------------------------
    # 7. Normalize
    # --------------------------------------------------------

    cls_gallery_n = F.normalize(
        cls_gallery,
        dim=1,
    )

    cls_query_n = F.normalize(
        cls_query,
        dim=1,
    )

    mean_gallery_n = F.normalize(
        mean_gallery,
        dim=1,
    )

    mean_query_n = F.normalize(
        mean_query,
        dim=1,
    )


    # --------------------------------------------------------
    # 8. Cosine retrieval
    # --------------------------------------------------------

    cls_similarity = (
        cls_query_n
        @ cls_gallery_n.T
    )

    mean_similarity = (
        mean_query_n
        @ mean_gallery_n.T
    )

    cls_topk_idx = torch.topk(
        cls_similarity,
        k=K,
        largest=True,
        dim=1,
    ).indices

    mean_topk_idx = torch.topk(
        mean_similarity,
        k=K,
        largest=True,
        dim=1,
    ).indices


    # --------------------------------------------------------
    # 9. Per-query evaluation
    # --------------------------------------------------------

    rows = []

    for qi, query_id in enumerate(
        query_ids
    ):

        query_categories = (
            image_categories[
                query_id
            ]
        )

        query_names = [
            category_names[c]
            for c in sorted(
                query_categories
            )
        ]

        strict_relevant_gallery = 0
        binary_relevant_gallery = 0

        for gallery_id in gallery_ids:

            relevance = pair_relevance(
                query_categories,
                image_categories[
                    gallery_id
                ],
            )

            binary_relevant_gallery += (
                relevance["binary"]
            )

            strict_relevant_gallery += (
                relevance["strict"]
            )


        cls_ids = [
            gallery_ids[index]
            for index
            in cls_topk_idx[
                qi
            ].tolist()
        ]

        mean_ids = [
            gallery_ids[index]
            for index
            in mean_topk_idx[
                qi
            ].tolist()
        ]


        cls_metrics = evaluate_topk(
            cls_ids,
            query_categories,
            image_categories,
        )

        mean_metrics = evaluate_topk(
            mean_ids,
            query_categories,
            image_categories,
        )


        row = {
            "query_id":
                query_id,

            "query_categories":
                str(
                    query_names
                ),

            "num_query_categories":
                len(
                    query_categories
                ),

            "binary_relevant_in_gallery":
                binary_relevant_gallery,

            "strict_relevant_in_gallery":
                strict_relevant_gallery,

            "cls_top5":
                str(
                    cls_ids
                ),

            "mean_patch_top5":
                str(
                    mean_ids
                ),

            "cls_binary_p5":
                cls_metrics[
                    "binary_p5"
                ],

            "mean_patch_binary_p5":
                mean_metrics[
                    "binary_p5"
                ],

            "cls_coverage_at5":
                cls_metrics[
                    "coverage_at5"
                ],

            "mean_patch_coverage_at5":
                mean_metrics[
                    "coverage_at5"
                ],

            "cls_jaccard_at5":
                cls_metrics[
                    "jaccard_at5"
                ],

            "mean_patch_jaccard_at5":
                mean_metrics[
                    "jaccard_at5"
                ],

            "cls_strict_hit5":
                cls_metrics[
                    "strict_hit5"
                ],

            "mean_patch_strict_hit5":
                mean_metrics[
                    "strict_hit5"
                ],

            "jaccard_cls_result":
                compare_pair(
                    cls_metrics[
                        "jaccard_at5"
                    ],
                    mean_metrics[
                        "jaccard_at5"
                    ],
                ),

            "coverage_cls_result":
                compare_pair(
                    cls_metrics[
                        "coverage_at5"
                    ],
                    mean_metrics[
                        "coverage_at5"
                    ],
                ),
        }

        rows.append(
            row
        )


    # --------------------------------------------------------
    # 10. Save per-query results
    # --------------------------------------------------------

    with open(
        PER_QUERY_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[
                0
            ].keys(),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


    # --------------------------------------------------------
    # 11. Eligibility
    # --------------------------------------------------------

    labeled_rows = [
        row
        for row in rows
        if (
            row[
                "num_query_categories"
            ]
            > 0
        )
    ]

    strict_eligible_rows = [
        row
        for row in labeled_rows
        if (
            row[
                "strict_relevant_in_gallery"
            ]
            > 0
        )
    ]


    # --------------------------------------------------------
    # 12. Paired counts
    # --------------------------------------------------------

    jaccard_results = [
        row[
            "jaccard_cls_result"
        ]
        for row in labeled_rows
    ]

    coverage_results = [
        row[
            "coverage_cls_result"
        ]
        for row in labeled_rows
    ]


    # --------------------------------------------------------
    # 13. Summary
    # --------------------------------------------------------

    summary = {
        "num_queries":
            len(rows),

        "num_labeled_queries":
            len(labeled_rows),

        "num_strict_eligible_queries":
            len(strict_eligible_rows),

        "cls_mean_binary_p5":
            mean(
                [
                    row[
                        "cls_binary_p5"
                    ]
                    for row
                    in labeled_rows
                ]
            ),

        "mean_patch_mean_binary_p5":
            mean(
                [
                    row[
                        "mean_patch_binary_p5"
                    ]
                    for row
                    in labeled_rows
                ]
            ),

        "cls_mean_coverage_at5":
            mean(
                [
                    row[
                        "cls_coverage_at5"
                    ]
                    for row
                    in labeled_rows
                ]
            ),

        "mean_patch_mean_coverage_at5":
            mean(
                [
                    row[
                        "mean_patch_coverage_at5"
                    ]
                    for row
                    in labeled_rows
                ]
            ),

        "cls_mean_jaccard_at5":
            mean(
                [
                    row[
                        "cls_jaccard_at5"
                    ]
                    for row
                    in labeled_rows
                ]
            ),

        "mean_patch_mean_jaccard_at5":
            mean(
                [
                    row[
                        "mean_patch_jaccard_at5"
                    ]
                    for row
                    in labeled_rows
                ]
            ),

        "cls_strict_hit5_eligible":
            mean(
                [
                    row[
                        "cls_strict_hit5"
                    ]
                    for row
                    in strict_eligible_rows
                ]
            ),

        "mean_patch_strict_hit5_eligible":
            mean(
                [
                    row[
                        "mean_patch_strict_hit5"
                    ]
                    for row
                    in strict_eligible_rows
                ]
            ),

        "jaccard_cls_wins":
            jaccard_results.count(
                "win"
            ),

        "jaccard_ties":
            jaccard_results.count(
                "tie"
            ),

        "jaccard_cls_losses":
            jaccard_results.count(
                "loss"
            ),

        "coverage_cls_wins":
            coverage_results.count(
                "win"
            ),

        "coverage_ties":
            coverage_results.count(
                "tie"
            ),

        "coverage_cls_losses":
            coverage_results.count(
                "loss"
            ),

        "cls_mean_patch_cosine_min":
            cls_mean_cosine.min().item(),

        "cls_mean_patch_cosine_mean":
            cls_mean_cosine.mean().item(),

        "cls_mean_patch_cosine_max":
            cls_mean_cosine.max().item(),
    }


    # --------------------------------------------------------
    # 14. Save summary
    # --------------------------------------------------------

    with open(
        SUMMARY_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=summary.keys(),
        )

        writer.writeheader()
        writer.writerow(
            summary
        )


    # --------------------------------------------------------
    # 15. Print summary
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "EXP024 VIT CLS VS MEAN-PATCH"
    )

    print(
        "=" * 70
    )

    for key, value in (
        summary.items()
    ):

        print(
            f"{key}: {value}"
        )

    print()

    print(
        "Saved:",
        PER_QUERY_PATH
    )

    print(
        "Saved:",
        SUMMARY_PATH
    )


if __name__ == "__main__":
    main()