from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def make_classifier(classifier):
    return ("classifier", classifier)


def smote():
    return ("smote", SMOTE())


def random_under_sampler():
    return ("random_under_sampler", RandomUnderSampler())
