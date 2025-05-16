from imblearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .utils import make_classifier, smote, random_under_sampler
from preprocessor import standard_preprocessor

pipes = []

pipes.append(Pipeline(steps=[
    standard_preprocessor(),
    make_classifier(XGBClassifier(eval_metric='aucpr', scale_pos_weight=0.136425648))
]))

pipes.append(make_pipeline(
    standard_preprocessor()[1],
    smote()[1],
    XGBClassifier(eval_metric='aucpr', scale_pos_weight=0.9)
))

pipes.append(make_pipeline(
    standard_preprocessor()[1],
    random_under_sampler()[1],
    XGBClassifier(eval_metric='aucpr', scale_pos_weight=0.136425648)
))
