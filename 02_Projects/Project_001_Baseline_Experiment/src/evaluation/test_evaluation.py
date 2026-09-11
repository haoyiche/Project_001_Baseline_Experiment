from src.evaluation.evaluate import evaluate



scores = [
    0.1,
    0.2,
    0.8,
    0.9
]


labels = [
    0,
    0,
    1,
    1
]



auc = evaluate(
    scores,
    labels
)


print(
    "AUROC:",
    auc
)