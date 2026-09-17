import pytest
from train import check_gate, run
import joblib


def test_gate_trips():
    acc = 0.99
    threshold = 1.0
    with pytest.raises(SystemExit) as exc_info:
        check_gate(acc, threshold)

    assert exc_info.value.code == 1


def test_gate_passes():
    acc = 1.0
    threshold = 0.99

    # If the check_gate doesn't raise any exception pytest will consider that as success test
    # Only fail when our model raises an exception
    # The absence of Exception is the assertion
    check_gate(acc, threshold)


def test_fail_run(tmp_path): # tmp_path is already a pathlib.Path object
    save_file = tmp_path / "iris_log_reg.joblib"
    with pytest.raises(SystemExit):
        run(save_file, 1.0)

    assert not save_file.exists()


def test_pass_run(tmp_path):

    save_file = tmp_path / "iris_log_reg.joblib"
    run(save_file, 0.0)
    assert save_file.exists()
    model = joblib.load(save_file)
    pred = model.predict([[5.1, 3.5, 1.4, 0.2]])[0]
    assert pred in (0, 1, 2)









