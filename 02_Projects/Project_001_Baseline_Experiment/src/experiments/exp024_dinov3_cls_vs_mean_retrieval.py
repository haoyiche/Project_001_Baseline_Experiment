import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModel,
)


# ============================================================
# Config
# ============================================================

MODEL_DIR = Path(
    r"D:\AI_Lab\models\dinov3"
    r"\dinov3-vits16-pretrain-lvd1689m"
)

COCO_ROOT = Path(
    "data/raw/coco2017"
)

IMAGE_DIR = (
    COCO_ROOT
    /  "val2017_subset500"
)

ANNOTATION_PATH = (
    COCO_ROOT
    / "annotations"
    / "instances_val2017.json"
)

OUTPUT_DIR = Path(
    "results/exp024/dinov3_cls_vs_mean_retrieval"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PER_QUERY_CSV = (
    OUTPUT_DIR
    / "per_query.csv"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "summary.csv"
)

SPLIT_CSV = (
    OUTPUT_DIR
    / "split.csv"
)


SEED = 42

NUM_IMAGES = 500
NUM_QUERY = 100
NUM_GALLERY = 400

TOP_K = 5

BATCH_SIZE = 8

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# COCO loading
# ============================================================

def load_coco_annotations(
    annotation_path,
):

    print(
        "Loading COCO annotations:",
        annotation_path
    )

    with open(
        annotation_path,
        "r",
        encoding="utf-8",
    ) as f:

        coco = json.load(
            f
        )


    # --------------------------------------------------------
    # image_id -> file_name
    # --------------------------------------------------------

    image_id_to_file = {}

    for image in coco[
        "images"
    ]:

        image_id = int(
            image["id"]
        )

        image_id_to_file[
            image_id
        ] = image[
            "file_name"
        ]


    # --------------------------------------------------------
    # image_id -> category set
    # --------------------------------------------------------

    image_id_to_categories = {
        image_id: set()
        for image_id
        in image_id_to_file.keys()
    }

    for annotation in coco[
        "annotations"
    ]:

        image_id = int(
            annotation[
                "image_id"
            ]
        )

        category_id = int(
            annotation[
                "category_id"
            ]
        )

        if image_id in (
            image_id_to_categories
        ):

            image_id_to_categories[
                image_id
            ].add(
                category_id
            )


    # --------------------------------------------------------
    # category_id -> category_name
    # --------------------------------------------------------

    category_id_to_name = {}

    for category in coco[
        "categories"
    ]:

        category_id_to_name[
            int(
                category["id"]
            )
        ] = category[
            "name"
        ]


    return (
        image_id_to_file,
        image_id_to_categories,
        category_id_to_name,
    )


# ============================================================
# Fixed split
# ============================================================

def build_fixed_split(
    image_id_to_file,
):

    valid_image_ids = []

    for image_id, file_name in (
        image_id_to_file.items()
    ):

        path = (
            IMAGE_DIR
            / file_name
        )

        if path.exists():

            valid_image_ids.append(
                image_id
            )


    valid_image_ids = sorted(
        valid_image_ids
    )

    assert (
        len(valid_image_ids)
        >= NUM_IMAGES
    ), (
        f"Need at least {NUM_IMAGES} images, "
        f"found {len(valid_image_ids)}"
    )


    rng = random.Random(
        SEED
    )

    sampled_ids = rng.sample(
        valid_image_ids,
        NUM_IMAGES,
    )


    # Keep split deterministic.
    #
    # First 100 sampled images:
    # query
    #
    # Remaining 400:
    # gallery

    query_ids = sampled_ids[
        :NUM_QUERY
    ]

    gallery_ids = sampled_ids[
        NUM_QUERY:
        NUM_QUERY
        + NUM_GALLERY
    ]


    assert (
        len(query_ids)
        == NUM_QUERY
    )

    assert (
        len(gallery_ids)
        == NUM_GALLERY
    )

    assert (
        set(query_ids)
        .isdisjoint(
            gallery_ids
        )
    )


    return (
        query_ids,
        gallery_ids,
    )


def save_split(
    query_ids,
    gallery_ids,
    image_id_to_file,
):

    rows = []

    for role, ids in [
        (
            "query",
            query_ids,
        ),
        (
            "gallery",
            gallery_ids,
        ),
    ]:

        for index, image_id in enumerate(
            ids
        ):

            rows.append(
                {
                    "role":
                        role,

                    "index":
                        index,

                    "image_id":
                        image_id,

                    "file_name":
                        image_id_to_file[
                            image_id
                        ],
                }
            )


    with open(
        SPLIT_CSV,
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


# ============================================================
# DINOv3 embedding extraction
# ============================================================

def load_images(
    image_ids,
    image_id_to_file,
):

    images = []

    for image_id in image_ids:

        path = (
            IMAGE_DIR
            / image_id_to_file[
                image_id
            ]
        )

        image = Image.open(
            path
        ).convert(
            "RGB"
        )

        images.append(
            image
        )

    return images


def extract_embeddings(
    image_ids,
    image_id_to_file,
    processor,
    model,
):

    all_cls = []
    all_mean_patch = []

    num_register_tokens = int(
        model.config.num_register_tokens
    )

    for start in range(
        0,
        len(image_ids),
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            len(image_ids),
        )

        batch_ids = image_ids[
            start:end
        ]

        images = load_images(
            batch_ids,
            image_id_to_file,
        )


        inputs = processor(
            images=images,
            return_tensors="pt",
        )

        pixel_values = (
            inputs[
                "pixel_values"
            ]
            .to(
                DEVICE
            )
        )


        with torch.inference_mode():

            outputs = model(
                pixel_values=pixel_values
            )


        hidden = (
            outputs
            .last_hidden_state
        )


        # ----------------------------------------------------
        # Sequence:
        #
        # [CLS]
        # [REGISTER x 4]
        # [PATCH x 196]
        #
        # at 224x224 for ViT-S/16
        # ----------------------------------------------------

        cls_embedding = hidden[
            :,
            0,
            :
        ]


        patch_tokens = hidden[
            :,
            1 + num_register_tokens:,
            :
        ]


        mean_patch_embedding = (
            patch_tokens.mean(
                dim=1
            )
        )


        # Normalize once here.
        #
        # Then cosine similarity becomes
        # plain dot product.

        cls_embedding = F.normalize(
            cls_embedding,
            p=2,
            dim=-1,
        )

        mean_patch_embedding = (
            F.normalize(
                mean_patch_embedding,
                p=2,
                dim=-1,
            )
        )


        all_cls.append(
            cls_embedding.cpu()
        )

        all_mean_patch.append(
            mean_patch_embedding.cpu()
        )


        print(
            "Extracted:",
            f"{end}/{len(image_ids)}"
        )


    all_cls = torch.cat(
        all_cls,
        dim=0,
    )

    all_mean_patch = torch.cat(
        all_mean_patch,
        dim=0,
    )


    assert (
        all_cls.shape[
            0
        ]
        == len(image_ids)
    )

    assert (
        all_mean_patch.shape
        == all_cls.shape
    )


    return (
        all_cls,
        all_mean_patch,
    )


# ============================================================
# Retrieval
# ============================================================

def topk_retrieval(
    query_embeddings,
    gallery_embeddings,
    k,
):

    # Both inputs are already L2-normalized.
    #
    # cosine similarity:
    #
    # q @ g.T

    similarity = (
        query_embeddings
        @ gallery_embeddings.T
    )


    topk_scores, topk_indices = (
        torch.topk(
            similarity,
            k=k,
            dim=1,
            largest=True,
            sorted=True,
        )
    )


    return (
        topk_indices,
        topk_scores,
    )


# ============================================================
# Metric helpers
# ============================================================

def binary_relevant(
    query_categories,
    gallery_categories,
):

    return (
        len(
            query_categories
            & gallery_categories
        )
        > 0
    )


def category_coverage(
    query_categories,
    gallery_categories,
):

    if (
        len(
            query_categories
        )
        == 0
    ):

        return float(
            "nan"
        )


    intersection = (
        query_categories
        & gallery_categories
    )


    return (
        len(
            intersection
        )
        / len(
            query_categories
        )
    )


def category_jaccard(
    query_categories,
    gallery_categories,
):

    union = (
        query_categories
        | gallery_categories
    )


    if (
        len(
            union
        )
        == 0
    ):

        return float(
            "nan"
        )


    intersection = (
        query_categories
        & gallery_categories
    )


    return (
        len(
            intersection
        )
        / len(
            union
        )
    )


def strict_relevant(
    query_categories,
    gallery_categories,
):

    if (
        len(
            query_categories
        )
        == 0
    ):

        return False


    return (
        query_categories
        .issubset(
            gallery_categories
        )
    )


def query_is_strict_eligible(
    query_categories,
    gallery_category_sets,
):

    if (
        len(
            query_categories
        )
        == 0
    ):

        return False


    return any(
        strict_relevant(
            query_categories,
            gallery_categories,
        )
        for gallery_categories
        in gallery_category_sets
    )


# ============================================================
# Per-query evaluation
# ============================================================

def evaluate_method(
    method_name,
    topk_indices,
    topk_scores,
    query_ids,
    gallery_ids,
    image_id_to_categories,
):

    gallery_category_sets = [
        image_id_to_categories[
            image_id
        ]
        for image_id
        in gallery_ids
    ]


    rows = []


    for query_index, query_id in enumerate(
        query_ids
    ):

        query_categories = (
            image_id_to_categories[
                query_id
            ]
        )


        labeled = (
            len(
                query_categories
            )
            > 0
        )


        eligible = (
            query_is_strict_eligible(
                query_categories,
                gallery_category_sets,
            )
        )


        retrieved_positions = (
            topk_indices[
                query_index
            ]
            .tolist()
        )


        retrieved_scores = (
            topk_scores[
                query_index
            ]
            .tolist()
        )


        retrieved_ids = [
            gallery_ids[
                position
            ]
            for position
            in retrieved_positions
        ]


        binary_values = []
        coverage_values = []
        jaccard_values = []
        strict_values = []


        for gallery_id in retrieved_ids:

            gallery_categories = (
                image_id_to_categories[
                    gallery_id
                ]
            )


            binary_values.append(
                float(
                    binary_relevant(
                        query_categories,
                        gallery_categories,
                    )
                )
            )


            coverage_values.append(
                category_coverage(
                    query_categories,
                    gallery_categories,
                )
            )


            jaccard_values.append(
                category_jaccard(
                    query_categories,
                    gallery_categories,
                )
            )


            strict_values.append(
                strict_relevant(
                    query_categories,
                    gallery_categories,
                )
            )


        if labeled:

            binary_p_at_k = float(
                np.mean(
                    binary_values
                )
            )

            coverage_at_k = float(
                np.mean(
                    coverage_values
                )
            )

            jaccard_at_k = float(
                np.mean(
                    jaccard_values
                )
            )

        else:

            binary_p_at_k = float(
                "nan"
            )

            coverage_at_k = float(
                "nan"
            )

            jaccard_at_k = float(
                "nan"
            )


        strict_hit_at_k = (
            float(
                any(
                    strict_values
                )
            )
            if eligible
            else float(
                "nan"
            )
        )


        rows.append(
            {
                "method":
                    method_name,

                "query_index":
                    query_index,

                "query_id":
                    query_id,

                "labeled":
                    labeled,

                "strict_eligible":
                    eligible,

                "binary_p_at_5":
                    binary_p_at_k,

                "coverage_at_5":
                    coverage_at_k,

                "jaccard_at_5":
                    jaccard_at_k,

                "strict_hit_at_5":
                    strict_hit_at_k,

                "top1_score":
                    float(
                        retrieved_scores[
                            0
                        ]
                    ),

                "retrieved_ids":
                    ";".join(
                        str(
                            image_id
                        )
                        for image_id
                        in retrieved_ids
                    ),
            }
        )


    return rows


# ============================================================
# Summary helpers
# ============================================================

def finite_values(
    rows,
    key,
):

    values = []

    for row in rows:

        value = row[
            key
        ]

        if np.isfinite(
            value
        ):

            values.append(
                float(
                    value
                )
            )


    return np.asarray(
        values,
        dtype=np.float64,
    )


def mean_metric(
    rows,
    key,
):

    values = finite_values(
        rows,
        key,
    )


    if len(
        values
    ) == 0:

        return float(
            "nan"
        )


    return float(
        values.mean()
    )


def paired_wtl(
    values_a,
    values_b,
    eps=1e-12,
):

    assert (
        len(
            values_a
        )
        == len(
            values_b
        )
    )


    wins = 0
    ties = 0
    losses = 0


    for a, b in zip(
        values_a,
        values_b,
    ):

        delta = (
            a
            - b
        )


        if delta > eps:

            wins += 1

        elif delta < -eps:

            losses += 1

        else:

            ties += 1


    return (
        wins,
        ties,
        losses,
    )


def aligned_metric_values(
    rows_a,
    rows_b,
    key,
):

    assert (
        len(
            rows_a
        )
        == len(
            rows_b
        )
    )


    values_a = []
    values_b = []


    for row_a, row_b in zip(
        rows_a,
        rows_b,
    ):

        assert (
            row_a[
                "query_id"
            ]
            == row_b[
                "query_id"
            ]
        )


        a = row_a[
            key
        ]

        b = row_b[
            key
        ]


        if (
            np.isfinite(
                a
            )
            and np.isfinite(
                b
            )
        ):

            values_a.append(
                float(
                    a
                )
            )

            values_b.append(
                float(
                    b
                )
            )


    return (
        np.asarray(
            values_a
        ),

        np.asarray(
            values_b
        ),
    )


# ============================================================
# Save per-query comparison
# ============================================================

def save_per_query(
    cls_rows,
    mean_rows,
):

    rows = []


    for cls_row, mean_row in zip(
        cls_rows,
        mean_rows,
    ):

        assert (
            cls_row[
                "query_id"
            ]
            == mean_row[
                "query_id"
            ]
        )


        query_id = (
            cls_row[
                "query_id"
            ]
        )


        row = {
            "query_index":
                cls_row[
                    "query_index"
                ],

            "query_id":
                query_id,

            "labeled":
                cls_row[
                    "labeled"
                ],

            "strict_eligible":
                cls_row[
                    "strict_eligible"
                ],

            "cls_binary_p5":
                cls_row[
                    "binary_p_at_5"
                ],

            "mean_binary_p5":
                mean_row[
                    "binary_p_at_5"
                ],

            "cls_coverage":
                cls_row[
                    "coverage_at_5"
                ],

            "mean_coverage":
                mean_row[
                    "coverage_at_5"
                ],

            "delta_coverage_cls_minus_mean":
                (
                    cls_row[
                        "coverage_at_5"
                    ]
                    - mean_row[
                        "coverage_at_5"
                    ]
                    if (
                        np.isfinite(
                            cls_row[
                                "coverage_at_5"
                            ]
                        )
                        and np.isfinite(
                            mean_row[
                                "coverage_at_5"
                            ]
                        )
                    )
                    else float(
                        "nan"
                    )
                ),

            "cls_jaccard":
                cls_row[
                    "jaccard_at_5"
                ],

            "mean_jaccard":
                mean_row[
                    "jaccard_at_5"
                ],

            "delta_jaccard_cls_minus_mean":
                (
                    cls_row[
                        "jaccard_at_5"
                    ]
                    - mean_row[
                        "jaccard_at_5"
                    ]
                    if (
                        np.isfinite(
                            cls_row[
                                "jaccard_at_5"
                            ]
                        )
                        and np.isfinite(
                            mean_row[
                                "jaccard_at_5"
                            ]
                        )
                    )
                    else float(
                        "nan"
                    )
                ),

            "cls_strict_hit":
                cls_row[
                    "strict_hit_at_5"
                ],

            "mean_strict_hit":
                mean_row[
                    "strict_hit_at_5"
                ],

            "cls_top1_score":
                cls_row[
                    "top1_score"
                ],

            "mean_top1_score":
                mean_row[
                    "top1_score"
                ],

            "cls_retrieved_ids":
                cls_row[
                    "retrieved_ids"
                ],

            "mean_retrieved_ids":
                mean_row[
                    "retrieved_ids"
                ],
        }


        rows.append(
            row
        )


    with open(
        PER_QUERY_CSV,
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


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Device:",
        DEVICE
    )

    print(
        "Seed:",
        SEED
    )


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    assert (
        IMAGE_DIR.exists()
    ), (
        f"Missing image dir: "
        f"{IMAGE_DIR}"
    )

    assert (
        ANNOTATION_PATH.exists()
    ), (
        f"Missing annotation: "
        f"{ANNOTATION_PATH}"
    )


    (
        image_id_to_file,
        image_id_to_categories,
        category_id_to_name,
    ) = load_coco_annotations(
        ANNOTATION_PATH
    )


    (
        query_ids,
        gallery_ids,
    ) = build_fixed_split(
        image_id_to_file
    )


    print()

    print(
        "Query images:",
        len(
            query_ids
        )
    )

    print(
        "Gallery images:",
        len(
            gallery_ids
        )
    )


    labeled_queries = sum(
        len(
            image_id_to_categories[
                image_id
            ]
        )
        > 0
        for image_id
        in query_ids
    )


    gallery_category_sets = [
        image_id_to_categories[
            image_id
        ]
        for image_id
        in gallery_ids
    ]


    strict_eligible_queries = sum(
        query_is_strict_eligible(
            image_id_to_categories[
                image_id
            ],
            gallery_category_sets,
        )
        for image_id
        in query_ids
    )


    print(
        "Labeled queries:",
        labeled_queries
    )

    print(
        "Strict eligible queries:",
        strict_eligible_queries
    )


    save_split(
        query_ids,
        gallery_ids,
        image_id_to_file,
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()

    print(
        "Loading DINOv3..."
    )


    processor = (
        AutoImageProcessor
        .from_pretrained(
            MODEL_DIR,
            local_files_only=True,
        )
    )


    model = (
        AutoModel
        .from_pretrained(
            MODEL_DIR,
            local_files_only=True,
        )
    )


    model.eval()

    model.to(
        DEVICE
    )


    hidden_size = int(
        model.config.hidden_size
    )

    patch_size = int(
        model.config.patch_size
    )

    num_register_tokens = int(
        model.config.num_register_tokens
    )


    print(
        "Hidden size:",
        hidden_size
    )

    print(
        "Patch size:",
        patch_size
    )

    print(
        "Register tokens:",
        num_register_tokens
    )


    # --------------------------------------------------------
    # Extract all 500 embeddings once
    # --------------------------------------------------------

    all_ids = (
        query_ids
        + gallery_ids
    )


    print()

    print(
        "=" * 80
    )

    print(
        "EXTRACT DINOv3 EMBEDDINGS"
    )

    print(
        "=" * 80
    )


    (
        all_cls,
        all_mean_patch,
    ) = extract_embeddings(
        all_ids,
        image_id_to_file,
        processor,
        model,
    )


    print()

    print(
        "CLS embeddings:",
        all_cls.shape
    )

    print(
        "Mean-patch embeddings:",
        all_mean_patch.shape
    )


    # --------------------------------------------------------
    # CLS vs mean-patch geometry
    # --------------------------------------------------------

    cls_mean_cos = (
        all_cls
        * all_mean_patch
    ).sum(
        dim=1
    )


    geometry = {
        "min":
            float(
                cls_mean_cos.min()
            ),

        "mean":
            float(
                cls_mean_cos.mean()
            ),

        "median":
            float(
                cls_mean_cos.median()
            ),

        "max":
            float(
                cls_mean_cos.max()
            ),

        "std":
            float(
                cls_mean_cos.std(
                    unbiased=True
                )
            ),
    }


    print()

    print(
        "=" * 80
    )

    print(
        "CLS VS MEAN-PATCH GEOMETRY"
    )

    print(
        "=" * 80
    )


    for key, value in (
        geometry.items()
    ):

        print(
            f"{key}: {value}"
        )


    # --------------------------------------------------------
    # Split embeddings
    # --------------------------------------------------------

    query_cls = all_cls[
        :NUM_QUERY
    ]

    gallery_cls = all_cls[
        NUM_QUERY:
    ]


    query_mean = all_mean_patch[
        :NUM_QUERY
    ]

    gallery_mean = all_mean_patch[
        NUM_QUERY:
    ]


    assert (
        query_cls.shape[
            0
        ]
        == NUM_QUERY
    )

    assert (
        gallery_cls.shape[
            0
        ]
        == NUM_GALLERY
    )


    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    (
        cls_topk_indices,
        cls_topk_scores,
    ) = topk_retrieval(
        query_cls,
        gallery_cls,
        TOP_K,
    )


    (
        mean_topk_indices,
        mean_topk_scores,
    ) = topk_retrieval(
        query_mean,
        gallery_mean,
        TOP_K,
    )


    # --------------------------------------------------------
    # Top-5 exact ranking overlap
    # --------------------------------------------------------

    exact_top5_match = (
        cls_topk_indices
        == mean_topk_indices
    ).all(
        dim=1
    )


    exact_top5_match_rate = float(
        exact_top5_match
        .float()
        .mean()
    )


    top5_set_overlap = []


    for query_index in range(
        NUM_QUERY
    ):

        cls_set = set(
            cls_topk_indices[
                query_index
            ].tolist()
        )

        mean_set = set(
            mean_topk_indices[
                query_index
            ].tolist()
        )


        overlap = (
            len(
                cls_set
                & mean_set
            )
            / TOP_K
        )


        top5_set_overlap.append(
            overlap
        )


    mean_top5_set_overlap = float(
        np.mean(
            top5_set_overlap
        )
    )


    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    cls_rows = evaluate_method(
        "cls",
        cls_topk_indices,
        cls_topk_scores,
        query_ids,
        gallery_ids,
        image_id_to_categories,
    )


    mean_rows = evaluate_method(
        "mean_patch",
        mean_topk_indices,
        mean_topk_scores,
        query_ids,
        gallery_ids,
        image_id_to_categories,
    )


    # --------------------------------------------------------
    # Mean metrics
    # --------------------------------------------------------

    cls_binary_p5 = mean_metric(
        cls_rows,
        "binary_p_at_5",
    )

    mean_binary_p5 = mean_metric(
        mean_rows,
        "binary_p_at_5",
    )


    cls_coverage = mean_metric(
        cls_rows,
        "coverage_at_5",
    )

    mean_coverage = mean_metric(
        mean_rows,
        "coverage_at_5",
    )


    cls_jaccard = mean_metric(
        cls_rows,
        "jaccard_at_5",
    )

    mean_jaccard = mean_metric(
        mean_rows,
        "jaccard_at_5",
    )


    cls_strict_hit = mean_metric(
        cls_rows,
        "strict_hit_at_5",
    )

    mean_strict_hit = mean_metric(
        mean_rows,
        "strict_hit_at_5",
    )


    # --------------------------------------------------------
    # Paired W/T/L
    # CLS relative to mean-patch
    # --------------------------------------------------------

    (
        cls_cov_values,
        mean_cov_values,
    ) = aligned_metric_values(
        cls_rows,
        mean_rows,
        "coverage_at_5",
    )


    (
        coverage_wins,
        coverage_ties,
        coverage_losses,
    ) = paired_wtl(
        cls_cov_values,
        mean_cov_values,
    )


    (
        cls_jac_values,
        mean_jac_values,
    ) = aligned_metric_values(
        cls_rows,
        mean_rows,
        "jaccard_at_5",
    )


    (
        jaccard_wins,
        jaccard_ties,
        jaccard_losses,
    ) = paired_wtl(
        cls_jac_values,
        mean_jac_values,
    )


    # --------------------------------------------------------
    # Save per-query
    # --------------------------------------------------------

    save_per_query(
        cls_rows,
        mean_rows,
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "seed":
            SEED,

        "num_images":
            NUM_IMAGES,

        "num_query":
            NUM_QUERY,

        "num_gallery":
            NUM_GALLERY,

        "top_k":
            TOP_K,

        "labeled_queries":
            labeled_queries,

        "strict_eligible_queries":
            strict_eligible_queries,

        "hidden_size":
            hidden_size,

        "patch_size":
            patch_size,

        "num_register_tokens":
            num_register_tokens,

        "cls_mean_cos_min":
            geometry[
                "min"
            ],

        "cls_mean_cos_mean":
            geometry[
                "mean"
            ],

        "cls_mean_cos_median":
            geometry[
                "median"
            ],

        "cls_mean_cos_max":
            geometry[
                "max"
            ],

        "cls_mean_cos_std":
            geometry[
                "std"
            ],

        "cls_binary_p5":
            cls_binary_p5,

        "mean_binary_p5":
            mean_binary_p5,

        "cls_coverage":
            cls_coverage,

        "mean_coverage":
            mean_coverage,

        "cls_jaccard":
            cls_jaccard,

        "mean_jaccard":
            mean_jaccard,

        "cls_strict_hit_at_5":
            cls_strict_hit,

        "mean_strict_hit_at_5":
            mean_strict_hit,

        "coverage_cls_wins":
            coverage_wins,

        "coverage_ties":
            coverage_ties,

        "coverage_cls_losses":
            coverage_losses,

        "jaccard_cls_wins":
            jaccard_wins,

        "jaccard_ties":
            jaccard_ties,

        "jaccard_cls_losses":
            jaccard_losses,

        "cls_mean_exact_top5_match_rate":
            exact_top5_match_rate,

        "cls_mean_mean_top5_set_overlap":
            mean_top5_set_overlap,
    }


    with open(
        SUMMARY_CSV,
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
    # Print final report
    # --------------------------------------------------------

    print()

    print(
        "=" * 80
    )

    print(
        "DINOv3 CLS VS MEAN-PATCH RETRIEVAL"
    )

    print(
        "=" * 80
    )


    print()

    print(
        "Dataset:"
    )

    print(
        "  Query:",
        NUM_QUERY
    )

    print(
        "  Gallery:",
        NUM_GALLERY
    )

    print(
        "  Labeled query:",
        labeled_queries
    )

    print(
        "  Strict eligible:",
        strict_eligible_queries
    )


    print()

    print(
        "Geometry:"
    )

    print(
        "  min:",
        geometry["min"]
    )

    print(
        "  mean:",
        geometry["mean"]
    )

    print(
        "  median:",
        geometry["median"]
    )

    print(
        "  max:",
        geometry["max"]
    )

    print(
        "  std:",
        geometry["std"]
    )


    print()

    print(
        "CLS:"
    )

    print(
        "  Binary P@5:",
        cls_binary_p5
    )

    print(
        "  Coverage:",
        cls_coverage
    )

    print(
        "  Jaccard:",
        cls_jaccard
    )

    print(
        "  Strict Hit@5:",
        cls_strict_hit
    )


    print()

    print(
        "Mean-patch:"
    )

    print(
        "  Binary P@5:",
        mean_binary_p5
    )

    print(
        "  Coverage:",
        mean_coverage
    )

    print(
        "  Jaccard:",
        mean_jaccard
    )

    print(
        "  Strict Hit@5:",
        mean_strict_hit
    )


    print()

    print(
        "Paired Coverage CLS W/T/L:",
        (
            coverage_wins,
            coverage_ties,
            coverage_losses,
        )
    )

    print(
        "Paired Jaccard CLS W/T/L:",
        (
            jaccard_wins,
            jaccard_ties,
            jaccard_losses,
        )
    )


    print()

    print(
        "CLS vs mean retrieval:"
    )

    print(
        "  Exact Top5 ranking match rate:",
        exact_top5_match_rate
    )

    print(
        "  Mean Top5 set overlap:",
        mean_top5_set_overlap
    )


    print()

    print(
        "Saved split:",
        SPLIT_CSV
    )

    print(
        "Saved per-query:",
        PER_QUERY_CSV
    )

    print(
        "Saved summary:",
        SUMMARY_CSV
    )


if __name__ == "__main__":
    main()