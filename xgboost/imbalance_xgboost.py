import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from xgboost.focal_loss import Focal_Loss

## https://github.com/jhwjhw0123/Imbalance-XGBoost
class ImbalanceXGBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators=10,
        max_depth=10,
        learning_rate=0.3,
        objective="binary:logistic",
        special_objective=None,
        gamma=0.25,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        scale_pos_weight=3,
        eval_metric="logloss",
        early_stopping_rounds=50,
        focal_alpha=None,
        focal_gamma=None,
        verbosity=3,
        n_jobs=-1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.objective = objective
        self.special_objective = special_objective
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.classifier = None

        ## FL specific params
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        self.eval_list = []

    def fit(self, X, y):

        ## Check if X, y are correctly shaped and applies basic input checks
        X, y = check_X_y(X, y)  ## Now X, y are numpyarrays
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        self.model_initialization_param_dict = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "scale_pos_weight": self.scale_pos_weight,
            "verbosity": self.verbosity,
            "n_jobs": self.n_jobs,
        }

        if self.special_objective is None:
            ## This means that there should be string objetctive
            if self.objective is None:
                raise ValueError(
                    "Argument objective must have a string referencing to supported objective function"
                )
            ## https://machinelearningmastery.com/xgboost-for-imbalanced-classification/
            ## https://stats.stackexchange.com/questions/243207/what-is-the-proper-usage-of-scale-pos-weight-in-xgboost-for-imbalanced-datasets
            ## This helps ensure that +ve/minority class gradients are scaled up to have higher contribution
            if self.scale_pos_weight is None:
                self.scale_pos_weight = (
                    1 - self.y_
                ).sum() / self.y_.sum()  ## Count of negative examples/ count of +ve examples
                self.model_initialization_param_dict[
                    "scale_pos_weight"
                ] = self.scale_pos_weight
            else:
                self.model_initialization_param_dict[
                    "scale_pos_weight"
                ] = self.scale_pos_weight
        else:
            if self.special_objective == "binary:focal_unweighted":
                if self.focal_gamma is None:
                    raise ValueError(
                        f"Argument focal_gamma must have a value when objective is {self.special_objective}"
                    )

                self.objective = Focal_Loss(
                    self.focal_gamma, None, 0
                ).calculate_derivatives
            elif self.special_objective == "binary:focal_weighted":
                if self.focal_gamma is None or self.focal_alpha is None:
                    raise ValueError(
                        f"Argument focal_gamma, focal_alpha must have a value when objective is {self.special_objective}"
                    )
                self.objective = Focal_Loss(
                    self.focal_gamma, self.focal_alpha, 0
                ).calculate_derivatives
            else:
                raise ValueError(
                    f"The input special objective mode is not recognzed! Could only be binary:focal_unweighted or binary:focal_weighted but got {self.special_objective}"
                )
        self.model_initialization_param_dict["objective"] = self.objective

        classifier = XGBClassifier()
        classifier.set_params(**self.model_initialization_param_dict)
        print(
            f"Paramaters set during XGBoost object instantiation: {self.model_initialization_param_dict}"
        )

        ## Create Train and Validation set for the purpose of tracking model performance during training
        sub_X_train, sub_X_test, sub_y_train, sub_y_test = train_test_split(
            self.X_, self.y_, test_size=0.10, random_state=1, stratify=self.y_
        )

        self.model_fit_param_dict = {
            "eval_metric": self.eval_metric,
            "early_stopping_rounds": self.early_stopping_rounds,
            "eval_set": [(sub_X_train, sub_y_train), (sub_X_test, sub_y_test)],
            "verbose": True,
        }

        classifier.fit(sub_X_train, sub_y_train, **self.model_fit_param_dict)
        self.classifier_ = classifier

        return self

    def link_function_logit_to_probability(self, logit):
        return 1.0 / (1.0 + np.exp(-logit))

    def predict_proba(self, X, use_till_best_iteration=True):

        ## Returns 2 d array with probability etaimate of belonging to each class
        check_is_fitted(self)
        ## Ensure tht X is finally a  numpy array
        X = check_array(X)

        ## Because we are using version 0.9.1 of XGBoost, we do not have to provide iteration_range in predict and predict_proba methods
        if (self.special_objective is not None) and (
            self.special_objective
            in ["binary:focal_unweighted", "binary:focal_weighted"]
        ):
            print(
                f"Using link function to convert logit to prob because we are using special objective function: {self.objective}"
            )

            if use_till_best_iteration and isinstance(self.early_stopping_rounds, int):
                print(
                    f"Using trees till best iteration: {self.classifier_.best_iteration} to make prediction because early_stopping rounds is set"
                )
                prob_or_logit = self.classifier_.predict(
                    X, output_margin=True, ntree_limit=self.classifier_.best_iteration
                )
            else:
                prob_or_logit = self.classifier_.predict(X, output_margin=True)

            ## prob_or_logit has logit. Hence convert it to prob
            prob_class_1 = self.link_function_logit_to_probability(prob_or_logit)
            return np.vstack([1 - prob_class_1, prob_class_1]).T

        else:
            ## In case, we are using predefined objective function, we can simply call inbuilt predict_proba
            print(
                f"Calling inbuilt predict_proba because we are using inbuilt objective function: {self.objective}"
            )
            if use_till_best_iteration and isinstance(self.early_stopping_rounds, int):
                print(
                    f"Using trees till iteration: {self.classifier_.best_iteration} to make prediction because early_stopping_rounds is set"
                )
                return self.classifier_.predict_proba(
                    X, ntree_limit=self.classifier_.best_iteration
                )
            else:
                return self.classifier_.predict_proba(X)
