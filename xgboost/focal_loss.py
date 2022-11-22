import numpy as np
from typing import Tuple


class Focal_Loss:

    ## https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html for custom objective function and/or evaluation metric
    def __init__(self, gamma, alpha, label_smooth):
        self.gamma = gamma
        self.alpha = alpha
        self.label_smooth = label_smooth

    def robust_pow(self, base, pow):
        return np.power(base, pow)

    def perform_label_smoothing(self, labels):
        ## Perform label smoothing as per formula present in https://arxiv.org/abs/1906.02629
        class_count = len(np.unique(labels))
        smoothed_labels = (
            1 - self.label_smooth
        ) * labels + self.label_smooth / class_count
        return smoothed_labels

    def calculate_derivatives(
        self, label: np.ndarray, predt: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        ## Note: because we are passing custom objective (focal loss), we will receive log odds in predt
        ## predt has to be converted into probability by passing it through sigmid function
        ## label: GT

        gamma = self.gamma
        alpha = self.alpha
        label = self.perform_label_smoothing(label)
        sigmoid_pred = 1.0 / (1.0 + np.exp(-predt))

        if alpha is None:
            n1 = label * self.robust_pow(1 - sigmoid_pred, gamma)
            n2 = (1 - label) * self.robust_pow(sigmoid_pred, gamma)
        else:
            n1 = label * alpha * self.robust_pow(1 - sigmoid_pred, gamma)
            n2 = (1 - label) * (1 - alpha) * self.robust_pow(sigmoid_pred, gamma)

        n3 = 1 - sigmoid_pred - (sigmoid_pred * gamma * np.log(sigmoid_pred + 1e-9))
        n4 = sigmoid_pred - (
            (1 - sigmoid_pred) * gamma * np.log(1 - sigmoid_pred + 1e-9)
        )

        gradient = n2 * n4 - n1 * n3
        hessian_part1 = n1 * (
            1
            + gamma * (1 + np.log(sigmoid_pred + 1e-9))
            + (gamma * n3) / (1 - sigmoid_pred)
        )
        hessian_part2 = n2 * (
            1
            + gamma * (1 + np.log(1 - sigmoid_pred + 1e-9))
            + (gamma * n4) / sigmoid_pred
        )
        hessian = sigmoid_pred * (1 - sigmoid_pred) * (hessian_part1 + hessian_part2)

        return gradient, hessian
