from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import logging
import joblib
import sys
from pathlib import Path

ACC_THRESHOLD = 0.88
# get the parent path of train to make sure we save at the correct dir even if we run from anywhere
DEFAULT_PATH = Path(__file__).resolve().parent / "iris_log_reg.joblib"
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def check_gate(acc, threshold=ACC_THRESHOLD):
    if acc <= threshold:
        sys.exit(1)


def run(model_path=DEFAULT_PATH, threshold=ACC_THRESHOLD):
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    logger.info(" %.3f ", acc)

    check_gate(acc, threshold)

    # save the model only if it passes the threshold

    joblib.dump(model, model_path)
    return acc


if __name__ == '__main__':

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)

    stream_formatter = logging.Formatter(fmt=" %(asctime)s | %(name)s | %(levelname)s | %(message)s ")
    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(stream_handler)

    run()

















