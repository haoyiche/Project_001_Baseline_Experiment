import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)


MANIFEST_PATH = Path(
    "configs/exp023_coco_subset500_ids.json"
)

ANNOTATION_PATH = Path(
    "data/raw/coco2017/annotations/instances_val2017.json"
)

IMAGE_DIR = Path(
    "data/raw/coco2017/val2017_subset500"
)

OUT_DIR = Path(
    "results/exp024"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PER_QUERY_PATH = (
    OUT_DIR
    / "retrieval_quantitative_v2_per_query.csv"
)

SUMMARY_PATH = (
    OUT_DIR
    / "retrieval_quantitative_v2_summary.csv"
)


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

NUM_GALLERY = 400
NUM_QUERY = 100
K = 5
BATCH_SIZE = 32


def image_path_from_id(image_id):
    return (
        IMAGE_DIR
        / f"{image_id:012d}.jpg"
    )


def mean(values):
    return (
        sum(values) / len(values)
        if values
        else float("nan")
    )


def extract_embeddings(
    image_ids,
    model,
    transform,
):

    all_embeddings = []

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

        batch = torch.stack(
            tensors,
            dim=0,
        ).to(
            DEVICE
        )

        with torch.no_grad():

            embeddings = model(
                batch
            )

        all_embeddings.append(
            embeddings.cpu()
        )

        print(
            f"Extracted "
            f"{min(start + BATCH_SIZE, len(image_ids))}"
            f"/{len(image_ids)}"
        )

    return torch.cat(
        all_embeddings,
        dim=0,
    )


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

    pair_metrics = []

    for gallery_id in retrieved_ids:

        pair_metrics.append(
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
                m["binary"]
                for m in pair_metrics
            ]
        ),

        "coverage_at5": mean(
            [
                m["coverage"]
                for m in pair_metrics
            ]
        ),

        "jaccard_at5": mean(
            [
                m["jaccard"]
                for m in pair_metrics
            ]
        ),

        "strict_hit5": int(
            any(
                m["strict"]
                for m in pair_metrics
            )
        ),
    }


def compare_pair(
    cosine_value,
    raw_value,
    eps=1e-12,
):

    delta = (
        cosine_value
        - raw_value
    )

    if delta > eps:
        return "win"

    if delta < -eps:
        return "loss"

    return "tie"


def main():

    # --------------------------------
    # Fixed image split
    # --------------------------------

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


    # --------------------------------
    # COCO category sets
    # --------------------------------

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


    # --------------------------------
    # Backbone
    # --------------------------------

    weights = (
        ResNet18_Weights.DEFAULT
    )

    transform = (
        weights.transforms()
    )

    model = resnet18(
        weights=weights
    )

    model.fc = (
        torch.nn.Identity()
    )

    model.eval()
    model.to(
        DEVICE
    )

    print(
        "Device:",
        DEVICE
    )


    # --------------------------------
    # Embeddings
    # --------------------------------

    all_eval_ids = (
        gallery_ids
        + query_ids
    )

    embeddings = (
        extract_embeddings(
            all_eval_ids,
            model,
            transform,
        )
    )

    gallery_embeddings = embeddings[
        :NUM_GALLERY
    ]

    query_embeddings = embeddings[
        NUM_GALLERY:
    ]

    print(
        "Gallery embedding shape:",
        gallery_embeddings.shape
    )

    print(
        "Query embedding shape:",
        query_embeddings.shape
    )


    # --------------------------------
    # Raw L2
    # --------------------------------

    raw_dist = torch.cdist(
        query_embeddings,
        gallery_embeddings,
        p=2,
    )

    raw_topk_idx = torch.topk(
        raw_dist,
        k=K,
        largest=False,
        dim=1,
    ).indices


    # --------------------------------
    # Cosine
    # --------------------------------

    gallery_n = F.normalize(
        gallery_embeddings,
        dim=1,
    )

    query_n = F.normalize(
        query_embeddings,
        dim=1,
    )

    cosine_sim = (
        query_n
        @ gallery_n.T
    )

    cosine_topk_idx = torch.topk(
        cosine_sim,
        k=K,
        largest=True,
        dim=1,
    ).indices


    # --------------------------------
    # Normalized L2 sanity check
    # --------------------------------

    normalized_l2 = torch.cdist(
        query_n,
        gallery_n,
        p=2,
    )

    normalized_topk_idx = torch.topk(
        normalized_l2,
        k=K,
        largest=False,
        dim=1,
    ).indices

    cosine_normalized_match = (
        cosine_topk_idx
        == normalized_topk_idx
    ).all(
        dim=1
    )


    # --------------------------------
    # Per-query evaluation
    # --------------------------------

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

        binary_relevant_gallery = 0
        strict_relevant_gallery = 0

        for gallery_id in gallery_ids:

            gallery_categories = (
                image_categories[
                    gallery_id
                ]
            )

            relevance = (
                pair_relevance(
                    query_categories,
                    gallery_categories,
                )
            )

            binary_relevant_gallery += (
                relevance[
                    "binary"
                ]
            )

            strict_relevant_gallery += (
                relevance[
                    "strict"
                ]
            )


        raw_ids = [
            gallery_ids[index]
            for index
            in raw_topk_idx[
                qi
            ].tolist()
        ]

        cosine_ids = [
            gallery_ids[index]
            for index
            in cosine_topk_idx[
                qi
            ].tolist()
        ]


        raw_metrics = evaluate_topk(
            raw_ids,
            query_categories,
            image_categories,
        )

        cosine_metrics = evaluate_topk(
            cosine_ids,
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

            "raw_top5":
                str(
                    raw_ids
                ),

            "cosine_top5":
                str(
                    cosine_ids
                ),

            "raw_binary_p5":
                raw_metrics[
                    "binary_p5"
                ],

            "cosine_binary_p5":
                cosine_metrics[
                    "binary_p5"
                ],

            "raw_coverage_at5":
                raw_metrics[
                    "coverage_at5"
                ],

            "cosine_coverage_at5":
                cosine_metrics[
                    "coverage_at5"
                ],

            "raw_jaccard_at5":
                raw_metrics[
                    "jaccard_at5"
                ],

            "cosine_jaccard_at5":
                cosine_metrics[
                    "jaccard_at5"
                ],

            "raw_strict_hit5":
                raw_metrics[
                    "strict_hit5"
                ],

            "cosine_strict_hit5":
                cosine_metrics[
                    "strict_hit5"
                ],

            "coverage_pair_result":
                compare_pair(
                    cosine_metrics[
                        "coverage_at5"
                    ],
                    raw_metrics[
                        "coverage_at5"
                    ],
                ),

            "jaccard_pair_result":
                compare_pair(
                    cosine_metrics[
                        "jaccard_at5"
                    ],
                    raw_metrics[
                        "jaccard_at5"
                    ],
                ),

            "cosine_equals_normalized_l2_top5":
                bool(
                    cosine_normalized_match[
                        qi
                    ].item()
                ),
        }

        rows.append(
            row
        )


    # --------------------------------
    # Save per-query
    # --------------------------------

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


    # --------------------------------
    # Valid rows
    # --------------------------------

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

    binary_eligible_rows = [
        row
        for row in labeled_rows
        if (
            row[
                "binary_relevant_in_gallery"
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


    # --------------------------------
    # Paired counts
    # --------------------------------

    coverage_results = [
        row[
            "coverage_pair_result"
        ]
        for row in labeled_rows
    ]

    jaccard_results = [
        row[
            "jaccard_pair_result"
        ]
        for row in labeled_rows
    ]


    # --------------------------------
    # Summary
    # --------------------------------

    summary = {
        "num_queries":
            len(rows),

        "num_labeled_queries":
            len(labeled_rows),

        "num_binary_eligible_queries":
            len(binary_eligible_rows),

        "num_strict_eligible_queries":
            len(strict_eligible_rows),

        "raw_mean_binary_p5":
            mean(
                [
                    row[
                        "raw_binary_p5"
                    ]
                    for row in labeled_rows
                ]
            ),

        "cosine_mean_binary_p5":
            mean(
                [
                    row[
                        "cosine_binary_p5"
                    ]
                    for row in labeled_rows
                ]
            ),

        "raw_mean_coverage_at5":
            mean(
                [
                    row[
                        "raw_coverage_at5"
                    ]
                    for row in labeled_rows
                ]
            ),

        "cosine_mean_coverage_at5":
            mean(
                [
                    row[
                        "cosine_coverage_at5"
                    ]
                    for row in labeled_rows
                ]
            ),

        "raw_mean_jaccard_at5":
            mean(
                [
                    row[
                        "raw_jaccard_at5"
                    ]
                    for row in labeled_rows
                ]
            ),

        "cosine_mean_jaccard_at5":
            mean(
                [
                    row[
                        "cosine_jaccard_at5"
                    ]
                    for row in labeled_rows
                ]
            ),

        "raw_strict_hit5_eligible":
            mean(
                [
                    row[
                        "raw_strict_hit5"
                    ]
                    for row
                    in strict_eligible_rows
                ]
            ),

        "cosine_strict_hit5_eligible":
            mean(
                [
                    row[
                        "cosine_strict_hit5"
                    ]
                    for row
                    in strict_eligible_rows
                ]
            ),

        "coverage_cosine_wins":
            coverage_results.count(
                "win"
            ),

        "coverage_ties":
            coverage_results.count(
                "tie"
            ),

        "coverage_cosine_losses":
            coverage_results.count(
                "loss"
            ),

        "jaccard_cosine_wins":
            jaccard_results.count(
                "win"
            ),

        "jaccard_ties":
            jaccard_results.count(
                "tie"
            ),

        "jaccard_cosine_losses":
            jaccard_results.count(
                "loss"
            ),

        "cosine_normalized_l2_top5_match_rate":
            mean(
                [
                    float(
                        row[
                            "cosine_equals_normalized_l2_top5"
                        ]
                    )
                    for row in rows
                ]
            ),
    }


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


    # --------------------------------
    # Print
    # --------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "EXP024 RETRIEVAL V2 SUMMARY"
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