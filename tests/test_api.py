import pytest
from fastapi.testclient import TestClient
from app.main import app, CLASSES


#TestClient only constructs an object. It wires up a transport that can deliver http events to our
# app . (it won't send a lifespan startup event) . because constructing a client is not same
# as booting up

# TestClient implements the context manager protocol precisely to fill that gap.
# __enter__ sends the startup event; __exit__ sends shutdown. So:


@pytest.fixture(scope='session')
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    # the client job is to hand us a response object and we will do assertion for status_code
    # and json()
    assert response.status_code == 200 # status = ok (200)
    assert response.json() == {'status' : 'ok'}


def test_predict(client):
    response = client.post('/predict', json={"sepal_length": 1.0, "sepal_width": 1.4, "petal_length": 2.3,"petal_width": 2.1})
    assert response.status_code == 200
    pred = response.json()
    assert CLASSES[pred['class_idx']] == pred['class_name']


# HTTP Code for failing pydantic validation
# 422 -> Unprocessable Entity (request parsed fine and syntax is valid but failed pydantic validation)
def test_invalid_pred(client):
    response = client.post('/predict', json={"sepal_length": 'hello', "sepal_width": 1.4, "petal_length": 2.3,"petal_width": 2.1})
    assert response.status_code == 422
    print(response.json())

# run with pytest -s (pytest swallows stdout by default; -s lets it through).
# You'll see a detail key holding a list of error objects, each naming the field location,
# the error type, and a message. That's the structure FastAPI generates from Pydantic's validation
# errors.


def test_int_coercion(client):
    response = client.post('/predict', json={"sepal_length": 5, "sepal_width": 1.4, "petal_length": 2.3,"petal_width": 2.1})
    assert response.status_code == 200


