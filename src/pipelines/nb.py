from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from .utils import make_classifier
from preprocessor import standard_preprocessor

pipes = []

pipes.append(Pipeline(steps=[
    standard_preprocessor(),
    make_classifier(GaussianNB())
]))
