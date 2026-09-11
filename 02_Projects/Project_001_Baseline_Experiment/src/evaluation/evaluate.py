import numpy as np
from sklearn.metrics import roc_auc_score



def evaluate(
        scores,
        labels
):

    scores = np.array(scores)

    labels = np.array(labels)


    auc = roc_auc_score(
        labels,
        scores
    )


    return auc