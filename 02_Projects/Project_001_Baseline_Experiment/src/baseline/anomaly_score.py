import torch


class AnomalyScorer:


    def __init__(self, center):

        self.center = center



    def score(self, features):

        distance = torch.norm(
            features - self.center,
            dim=1
        )

        return distance