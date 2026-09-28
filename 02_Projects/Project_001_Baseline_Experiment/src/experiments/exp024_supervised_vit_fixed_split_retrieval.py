import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from torchvision.models import (
    ViT_B_16_Weights,
    vit_b_16,
)

from exp024_dinov3_cls_vs_mean_retrieval import (
    load_coco_annotations,
    topk_retrieval,
    evaluate_method,
    mean_metric,
    aligned_metric_values,
    paired_wtl,
)


# ============================================================
# Config
# ============================================================

COCO_ROOT = Path(
    "data/raw/coco2017"
)

IMAGE_DIR = (
    COCO_ROOT
    / "val2017_subset500"
)

ANNOTATION_PATH = (
    COCO_ROOT
    / "annotations"
    / "instances_val2017.json"
)

FIXED_SPLIT_PATH = Path(
    "results/exp024/"
    "dinov3_cls_vs_mean_retrieval/"
    "split.csv"
)

OUTPUT_DIR = Path(
    "results/exp024/"
    "supervised_vit_fixed_split_retrieval"
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

TOP_K = 5
BATCH_SIZE = 8

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Fixed split
# ============================================================

def load_fixed_split(
    split_path,
):

    query_rows = []
    gallery_rows = []

    with open(
        split_path,
        "r",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:

            item = {
                "index":
                    int(
                        row["index"]
                    ),

                "image_id":
                    int(
                        row["image_id"]
                    ),

                "file_name":
                    row["file_name"],
            }

            if (
                row["role"]
                == "query"
            ):

                query_rows.append(
                    item
                )

            elif (
                row["role"]
                == "gallery"
            ):

                gallery_rows.append(
                    item
                )

            else:

                raise ValueError(
                    f"Unknown role: "
                    f"{row['role']}"
                )


    query_rows = sorted(
        query_rows,
        key=lambda x: x[
            "index"
        ],
    )

    gallery_rows = sorted(
        gallery_rows,
        key=lambda x: x[
            "index"
        ],
    )


    query_ids = [
        row["image_id"]
        for row
        in query_rows
    ]

    gallery_ids = [
        row["image_id"]
        for row
        in gallery_rows
    ]


    assert len(
        query_ids
    ) == 100

    assert len(
        gallery_ids
    ) == 400

    assert set(
        query_ids
    ).isdisjoint(
        gallery_ids
    )


    return (
        query_ids,
        gallery_ids,
    )


# ============================================================
# Feature extraction
# ============================================================

def extract_embeddings(
    image_ids,
    image_id_to_file,
    model,
    transform,
):

    all_cls = []
    all_mean_patch = []


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

        tensors = []


        for image_id in batch_ids:

            image_path = (
                IMAGE_DIR
                / image_id_to_file[
                    image_id
                ]
            )

            assert (
                image_path.exists()
            ), image_path


            image = Image.open(
                image_path
            ).convert(
                "RGB"
            )


            tensor = transform(
                image
            )

            tensors.append(
                tensor
            )


        x = torch.stack(
            tensors,
            dim=0,
        ).to(
            DEVICE
        )


        with torch.inference_mode():

            # --------------------------------------------
            # torchvision ViT internals
            #
            # _process_input:
            # [B, 3, 224, 224]
            # ->
            # [B, 196, 768]
            # --------------------------------------------

            patches = model._process_input(
                x
            )

            batch_size = (
                patches.shape[
                    0
                ]
            )


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
                    patches,
                ],
                dim=1,
            )


            encoded = model.encoder(
                tokens
            )


            cls_embedding = encoded[
                :,
                0,
                :
            ]


            patch_tokens = encoded[
                :,
                1:,
                :
            ]


            mean_patch_embedding = (
                patch_tokens.mean(
                    dim=1
                )
            )


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


    return (
        all_cls,
        all_mean_patch,
    )


# ============================================================
# Per-query CSV
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
            cls_row["query_id"]
            == mean_row["query_id"]
        )


        rows.append(
            {
                "query_id":
                    cls_row[
                        "query_id"
                    ],

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

                "cls_jaccard":
                    cls_row[
                        "jaccard_at_5"
                    ],

                "mean_jaccard":
                    mean_row[
                        "jaccard_at_5"
                    ],

                "cls_strict_hit":
                    cls_row[
                        "strict_hit_at_5"
                    ],

                "mean_strict_hit":
                    mean_row[
                        "strict_hit_at_5"
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


    assert (
        FIXED_SPLIT_PATH.exists()
    ), FIXED_SPLIT_PATH


    # --------------------------------------------------------
    # Same exact split as DINOv3
    # --------------------------------------------------------

    (
        query_ids,
        gallery_ids,
    ) = load_fixed_split(
        FIXED_SPLIT_PATH
    )


    print(
        "Query:",
        len(
            query_ids
        )
    )

    print(
        "Gallery:",
        len(
            gallery_ids
        )
    )


    # --------------------------------------------------------
    # COCO metadata
    # --------------------------------------------------------

    (
        image_id_to_file,
        image_id_to_categories,
        _,
    ) = load_coco_annotations(
        ANNOTATION_PATH
    )


    gallery_category_sets = [
        image_id_to_categories[
            image_id
        ]
        for image_id
        in gallery_ids
    ]


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


    strict_eligible_queries = sum(
        (
            len(
                image_id_to_categories[
                    query_id
                ]
            )
            > 0
        )
        and any(
            image_id_to_categories[
                query_id
            ].issubset(
                gallery_categories
            )
            for gallery_categories
            in gallery_category_sets
        )
        for query_id
        in query_ids
    )


    print(
        "Labeled queries:",
        labeled_queries
    )

    print(
        "Strict eligible:",
        strict_eligible_queries
    )


    # Critical protocol assertion.
    #
    # DINOv3 fixed split produced:
    #
    # labeled = 99
    # strict eligible = 59

    assert (
        labeled_queries == 99
    )

    assert (
        strict_eligible_queries == 59
    )


    # --------------------------------------------------------
    # Supervised ViT
    # --------------------------------------------------------

    print()

    print(
        "Loading supervised "
        "ViT-B/16..."
    )


    weights = (
        ViT_B_16_Weights.DEFAULT
    )


    model = vit_b_16(
        weights=weights
    )


    model.eval()

    model.to(
        DEVICE
    )


    transform = (
        weights.transforms()
    )


    print(
        "Hidden dim:",
        model.hidden_dim
    )

    print(
        "Patch size:",
        model.patch_size
    )


    # --------------------------------------------------------
    # Extract same 500 images in exact split order
    # --------------------------------------------------------

    all_ids = (
        query_ids
        + gallery_ids
    )


    (
        all_cls,
        all_mean,
    ) = extract_embeddings(
        all_ids,
        image_id_to_file,
        model,
        transform,
    )


    print()

    print(
        "CLS:",
        all_cls.shape
    )

    print(
        "Mean patch:",
        all_mean.shape
    )


    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    cls_mean_cos = (
        all_cls
        * all_mean
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
                cls_mean_cos.std()
            ),
    }


    # --------------------------------------------------------
    # Fixed split embeddings
    # --------------------------------------------------------

    num_query = len(
        query_ids
    )


    query_cls = all_cls[
        :num_query
    ]

    gallery_cls = all_cls[
        num_query:
    ]


    query_mean = all_mean[
        :num_query
    ]

    gallery_mean = all_mean[
        num_query:
    ]


    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    (
        cls_indices,
        cls_scores,
    ) = topk_retrieval(
        query_cls,
        gallery_cls,
        TOP_K,
    )


    (
        mean_indices,
        mean_scores,
    ) = topk_retrieval(
        query_mean,
        gallery_mean,
        TOP_K,
    )


    cls_rows = evaluate_method(
        "cls",
        cls_indices,
        cls_scores,
        query_ids,
        gallery_ids,
        image_id_to_categories,
    )


    mean_rows = evaluate_method(
        "mean_patch",
        mean_indices,
        mean_scores,
        query_ids,
        gallery_ids,
        image_id_to_categories,
    )


    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    cls_binary = mean_metric(
        cls_rows,
        "binary_p_at_5",
    )

    mean_binary = mean_metric(
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


    cls_strict = mean_metric(
        cls_rows,
        "strict_hit_at_5",
    )

    mean_strict = mean_metric(
        mean_rows,
        "strict_hit_at_5",
    )


    # --------------------------------------------------------
    # Paired W/T/L
    # --------------------------------------------------------

    cls_cov, mean_cov = (
        aligned_metric_values(
            cls_rows,
            mean_rows,
            "coverage_at_5",
        )
    )


    cov_wtl = paired_wtl(
        cls_cov,
        mean_cov,
    )


    cls_jac, mean_jac = (
        aligned_metric_values(
            cls_rows,
            mean_rows,
            "jaccard_at_5",
        )
    )


    jac_wtl = paired_wtl(
        cls_jac,
        mean_jac,
    )


    # --------------------------------------------------------
    # Top-5 neighborhood overlap
    # --------------------------------------------------------

    exact_match = (
        cls_indices
        == mean_indices
    ).all(
        dim=1
    )


    exact_match_rate = float(
        exact_match.float().mean()
    )


    set_overlaps = []


    for i in range(
        num_query
    ):

        cls_set = set(
            cls_indices[
                i
            ].tolist()
        )

        mean_set = set(
            mean_indices[
                i
            ].tolist()
        )


        set_overlaps.append(
            len(
                cls_set
                & mean_set
            )
            / TOP_K
        )


    mean_set_overlap = float(
        np.mean(
            set_overlaps
        )
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_per_query(
        cls_rows,
        mean_rows,
    )


    summary = {
        "labeled_queries":
            labeled_queries,

        "strict_eligible_queries":
            strict_eligible_queries,

        "cls_mean_cos_min":
            geometry["min"],

        "cls_mean_cos_mean":
            geometry["mean"],

        "cls_mean_cos_median":
            geometry["median"],

        "cls_mean_cos_max":
            geometry["max"],

        "cls_mean_cos_std":
            geometry["std"],

        "cls_binary_p5":
            cls_binary,

        "mean_binary_p5":
            mean_binary,

        "cls_coverage":
            cls_coverage,

        "mean_coverage":
            mean_coverage,

        "cls_jaccard":
            cls_jaccard,

        "mean_jaccard":
            mean_jaccard,

        "cls_strict_hit":
            cls_strict,

        "mean_strict_hit":
            mean_strict,

        "coverage_cls_wins":
            cov_wtl[0],

        "coverage_ties":
            cov_wtl[1],

        "coverage_cls_losses":
            cov_wtl[2],

        "jaccard_cls_wins":
            jac_wtl[0],

        "jaccard_ties":
            jac_wtl[1],

        "jaccard_cls_losses":
            jac_wtl[2],

        "exact_top5_match_rate":
            exact_match_rate,

        "mean_top5_set_overlap":
            mean_set_overlap,
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
    # Report
    # --------------------------------------------------------

    print()

    print(
        "=" * 80
    )

    print(
        "SUPERVISED ViT "
        "FIXED-SPLIT RETRIEVAL"
    )

    print(
        "=" * 80
    )


    print()

    print(
        "Geometry:"
    )

    for key, value in (
        geometry.items()
    ):

        print(
            f"  {key}: {value}"
        )


    print()

    print(
        "CLS:"
    )

    print(
        "  Binary P@5:",
        cls_binary
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
        cls_strict
    )


    print()

    print(
        "Mean-patch:"
    )

    print(
        "  Binary P@5:",
        mean_binary
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
        mean_strict
    )


    print()

    print(
        "Coverage CLS W/T/L:",
        cov_wtl
    )

    print(
        "Jaccard CLS W/T/L:",
        jac_wtl
    )


    print()

    print(
        "Exact Top5 match rate:",
        exact_match_rate
    )

    print(
        "Mean Top5 set overlap:",
        mean_set_overlap
    )


    print()

    print(
        "Saved:",
        PER_QUERY_CSV
    )

    print(
        "Saved:",
        SUMMARY_CSV
    )


if __name__ == "__main__":
    main()