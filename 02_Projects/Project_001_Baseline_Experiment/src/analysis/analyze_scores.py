def analyze_scores(
        scores,
        labels,
        paths,
        top_k=5
):


    results = list(
        zip(
            scores,
            labels,
            paths
        )
    )


    # =====================
    # Top anomaly
    # =====================

    top_anomaly = sorted(
        results,
        key=lambda x:x[0],
        reverse=True
    )[:top_k]



    # =====================
    # Hard defect
    # label=1 score最低
    # =====================

    defects = [
        r for r in results
        if r[1] == 1
    ]


    hard_defects = sorted(
        defects,
        key=lambda x:x[0]
    )[:top_k]



    # =====================
    # False positive
    # label=0 score最高
    # =====================

    normals = [
        r for r in results
        if r[1] == 0
    ]


    false_positive = sorted(
        normals,
        key=lambda x:x[0],
        reverse=True
    )[:top_k]


    return {

        "top_anomaly": top_anomaly,

        "hard_defect": hard_defects,

        "false_positive": false_positive

    }