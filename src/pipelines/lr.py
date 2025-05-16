from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .utils import make_classifier
from preprocessor import scalar, standard_preprocessor, numeric_to_bool, standard_cat_pipeline, standard_bool_pipeline, make_preprocessor
from data import num_features

pipes = []

pipes.append(Pipeline(steps=[
    make_preprocessor([
        standard_cat_pipeline(),
        standard_bool_pipeline(),
        ("numerical", Pipeline(
            steps=[
                scalar(),
            ]
        ), [f for f in num_features if f != "Previous Contact Days_log"]),
        ("special_numerical", Pipeline(
            steps=[
                numeric_to_bool()
            ]
        ), ["Previous Contact Days_log"]),
    ]),
    make_classifier(LogisticRegression(max_iter=500))
]))

pipes.append(Pipeline(steps=[
    standard_preprocessor(),
    make_classifier(LogisticRegression(max_iter=1500, class_weight="balanced"))
]))
