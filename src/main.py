import logging
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from logger import formatter
from data import X, y
from pipelines import lr_pipes, nb_pipes, nn_pipes, svm_pipes, tree_pipes

top = []
top_logger = logging.getLogger("top_" + __name__)
top_file_handler = logging.FileHandler(f"logs/top.log", mode="w")
top_file_handler.setFormatter(formatter)
top_logger.addHandler(top_file_handler)


def evaluate(pipe: Pipeline, logger: logging.Logger):
    # Log pipe structure
    struct_log = "[Pipeline Structure]"
    if hasattr(pipe, "named_steps"):
        for step_name, step in pipe.named_steps.items():
            struct_log += f"\n{step_name}: {step.__class__.__name__}"
            # Add configuration details if available
            if hasattr(step, "get_params"):
                params = step.get_params()
                param_log = ", ".join([f"{k}={v}" for k, v in params.items()])
                struct_log += f" ({param_log})"
    logger.info(struct_log)

    # Split data and fit pipe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)

    # Start collecting metrices
    log = "\n Metrices"

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    log += f'\nAccuracy: {accuracy:.4f}'

    # More detailed classification report (precision, recall, f1-score)
    log += f"\nClassification Report: \n{classification_report(y_test, y_pred)}"

    # ROC-AUC score (for binary classification)
    if hasattr(pipe, "predict_proba"):
        roc_auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
        log += f'\nROC-AUC: {roc_auc:.4f}'
    logger.info(log)

    recall = recall_score(y_test, y_pred, pos_label=1)
    precision = precision_score(y_test, y_pred, pos_label=1)
    return recall, precision


def run_evaluate(pipes, filename: str):
    logger = logging.getLogger(filename)
    file_handler = logging.FileHandler(f"logs/{filename}.log", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Logging {filename}")
    for idx, pipe in enumerate(pipes):
        recall, precision = evaluate(pipe, logger)
        magnitude = (recall + precision*1.5) / 2
        imbalance = abs(recall - precision)
        balance = 1 - imbalance
        score = magnitude * balance
        top.append((score, precision, recall, idx, filename))


run_evaluate(lr_pipes, "lr")
run_evaluate(tree_pipes, "tree")
run_evaluate(svm_pipes, "svm")
run_evaluate(nb_pipes, "nb")
run_evaluate(nn_pipes, "nn")

top_logger.info(f"\nTop 3:\n{sorted(top)[-3:]}")
