import ast
import csv
import json
from collections import defaultdict
from pathlib import Path


MANIFEST_PATH = Path(
    "configs/exp023_coco_subset500_ids.json"
)

ANNOTATION_PATH = Path(
    "data/raw/coco2017/annotations/instances_val2017.json"
)

RETRIEVAL_PATH = Path(
    "results/exp024/retrieval_minimal.csv"
)

OUTPUT_PATH = Path(
    "results/exp024/retrieval_quantitative.csv"
)

SUMMARY_PATH = Path(
    "results/exp024/retrieval_quantitative_summary.csv"
)

NUM_GALLERY = 80
K = 5


def mean(values):
    return (
        sum(values) / len(values)
        if values
        else float("nan")
    )


def evaluate_topk(
    retrieved_ids,
    query_categories,
    image_categories,
    relevant_in_gallery,
):
    flags = []

    for image_id in retrieved_ids:

        gallery_categories = (
            image_categories[image_id]
        )

        is_relevant = bool(
            query_categories
            & gallery_categories
        )

        flags.append(
            is_relevant
        )

    num_relevant_retrieved = sum(
        flags
    )

    precision_at_k = (
        num_relevant_retrieved
        / K
    )

    hit_at_k = int(
        num_relevant_retrieved > 0
    )

    if relevant_in_gallery > 0:
        recall_at_k = (
            num_relevant_retrieved
            / relevant_in_gallery
        )
    else:
        recall_at_k = float("nan")

    return {
        "num_relevant_retrieved": (
            num_relevant_retrieved
        ),
        "precision_at_5": (
            precision_at_k
        ),
        "hit_at_5": (
            hit_at_k
        ),
        "recall_at_5": (
            recall_at_k
        ),
    }


def main():

    # --------------------------------
    # Load fixed subset
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


    # --------------------------------
    # Load COCO annotations
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
        for category in coco[
            "categories"
        ]
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
    # Load retrieval results
    # --------------------------------

    with open(
        RETRIEVAL_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        retrieval_rows = list(
            csv.DictReader(f)
        )


    output_rows = []


    for row in retrieval_rows:

        query_id = int(
            row["query_id"]
        )

        query_categories = (
            image_categories[
                query_id
            ]
        )


        query_category_names = [
            category_names[c]
            for c in sorted(
                query_categories
            )
        ]


        relevant_gallery_ids = []

        for gallery_id in gallery_ids:

            if (
                query_categories
                & image_categories[
                    gallery_id
                ]
            ):
                relevant_gallery_ids.append(
                    gallery_id
                )


        relevant_in_gallery = len(
            relevant_gallery_ids
        )


        raw_ids = ast.literal_eval(
            row[
                "raw_l2_top5"
            ]
        )

        cosine_ids = ast.literal_eval(
            row[
                "cosine_top5"
            ]
        )


        raw_metrics = evaluate_topk(
            raw_ids,
            query_categories,
            image_categories,
            relevant_in_gallery,
        )


        cosine_metrics = (
            evaluate_topk(
                cosine_ids,
                query_categories,
                image_categories,
                relevant_in_gallery,
            )
        )


        result = {
            "query_id":
                query_id,

            "query_categories":
                str(
                    query_category_names
                ),

            "relevant_in_gallery":
                relevant_in_gallery,

            "raw_relevant_top5":
                raw_metrics[
                    "num_relevant_retrieved"
                ],

            "raw_precision_at_5":
                raw_metrics[
                    "precision_at_5"
                ],

            "raw_hit_at_5":
                raw_metrics[
                    "hit_at_5"
                ],

            "raw_recall_at_5":
                raw_metrics[
                    "recall_at_5"
                ],

            "cosine_relevant_top5":
                cosine_metrics[
                    "num_relevant_retrieved"
                ],

            "cosine_precision_at_5":
                cosine_metrics[
                    "precision_at_5"
                ],

            "cosine_hit_at_5":
                cosine_metrics[
                    "hit_at_5"
                ],

            "cosine_recall_at_5":
                cosine_metrics[
                    "recall_at_5"
                ],
        }


        output_rows.append(
            result
        )


        print()
        print(
            "=" * 70
        )

        print(
            "Query:",
            query_id
        )

        print(
            "Categories:",
            query_category_names
        )

        print(
            "Relevant in gallery:",
            relevant_in_gallery
        )

        print(
            "RAW    P@5:",
            raw_metrics[
                "precision_at_5"
            ],
        )

        print(
            "COSINE P@5:",
            cosine_metrics[
                "precision_at_5"
            ],
        )

        print(
            "RAW    Hit@5:",
            raw_metrics[
                "hit_at_5"
            ],
        )

        print(
            "COSINE Hit@5:",
            cosine_metrics[
                "hit_at_5"
            ],
        )


    # --------------------------------
    # Save per-query results
    # --------------------------------

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=output_rows[
                0
            ].keys(),
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )


    # --------------------------------
    # Summary
    # --------------------------------

    all_raw_p5 = [
        row[
            "raw_precision_at_5"
        ]
        for row in output_rows
    ]

    all_cos_p5 = [
        row[
            "cosine_precision_at_5"
        ]
        for row in output_rows
    ]


    eligible_rows = [
        row
        for row in output_rows
        if (
            row[
                "relevant_in_gallery"
            ]
            > 0
        )
    ]


    eligible_raw_p5 = [
        row[
            "raw_precision_at_5"
        ]
        for row in eligible_rows
    ]

    eligible_cos_p5 = [
        row[
            "cosine_precision_at_5"
        ]
        for row in eligible_rows
    ]


    summary = {
        "num_queries":
            len(output_rows),

        "num_eligible_queries":
            len(eligible_rows),

        "raw_mean_p5_all":
            mean(
                all_raw_p5
            ),

        "cosine_mean_p5_all":
            mean(
                all_cos_p5
            ),

        "raw_mean_p5_eligible":
            mean(
                eligible_raw_p5
            ),

        "cosine_mean_p5_eligible":
            mean(
                eligible_cos_p5
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


    print()
    print(
        "=" * 70
    )

    print(
        "SUMMARY"
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
        OUTPUT_PATH,
    )

    print(
        "Saved:",
        SUMMARY_PATH,
    )


if __name__ == "__main__":
    main()