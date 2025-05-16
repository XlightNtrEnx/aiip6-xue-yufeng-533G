from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from .utils import make_classifier
from preprocessor import standard_preprocessor

pipes = []

pipes.append(Pipeline(steps=[
    standard_preprocessor(),
    make_classifier(MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation="relu", solver="sgd", max_iter=1500))
]))

pipes.append(Pipeline(steps=[
    standard_preprocessor(),
    make_classifier(MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu", solver="sgd", max_iter=1500))
]))

pipes.append(Pipeline(steps=[
    standard_preprocessor(),
    make_classifier(MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="sgd", max_iter=1500))
]))
