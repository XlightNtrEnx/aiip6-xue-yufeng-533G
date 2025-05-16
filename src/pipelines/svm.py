from imblearn.pipeline import make_pipeline as make_imbpipeline, Pipeline as ImbPipeline
from sklearn.svm import SVC

from .utils import make_classifier, smote
from preprocessor import standard_preprocessor

pipes = []

pipes.append(make_imbpipeline(
    standard_preprocessor()[1],
    smote()[1],
    SVC(class_weight='balanced', random_state=42)
))

pipes.append(make_imbpipeline(
    standard_preprocessor()[1],
    SVC(class_weight='balanced', random_state=42)
))

pipes.append(make_imbpipeline(
    standard_preprocessor()[1],
    smote()[1],
    SVC(random_state=42)
))

pipes.append(make_imbpipeline(
    standard_preprocessor()[1],
    SVC(random_state=42)
))

pipes.append(ImbPipeline(steps=[
    standard_preprocessor(),
    smote(),
    make_classifier(SVC(random_state=42))
]))

pipes.append(ImbPipeline(steps=[
    standard_preprocessor(),
    make_classifier(SVC(random_state=42))
]))
