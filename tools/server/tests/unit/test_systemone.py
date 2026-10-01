import pytest
from utils import *

server = ServerPreset.tinylaya()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinylaya()


TEST_STATE = "I was charged twice for my order last week and nobody has replied."

TEST_QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "payments and refunds", "shipping": None, "technical": None},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["can wait", "this week", "today", "right now"],
    },
    "angry": {
        "type": "noul",
        "instructions": "Is the customer angry?",
    },
}


def test_systemone():
    global server
    server.start()
    res = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        "questions": TEST_QUESTIONS,
    })
    assert res.status_code == 200
    assert res.body["usage"]["input_tokens"] > 0
    assert res.body["usage"]["output_tokens"] == 0

    answers = res.body["answers"]
    assert list(answers.keys()) == ["route", "urgency", "angry"]

    route = answers["route"]
    assert route["type"] == "choice"
    assert list(route["probabilities"].keys()) == ["billing", "shipping", "technical"]
    assert abs(sum(route["probabilities"].values()) - 1.0) < 1e-4
    assert route["choice"] == max(route["probabilities"], key=route["probabilities"].get)
    assert 0.0 <= route["confidence"] <= 1.0

    urgency = answers["urgency"]
    assert urgency["type"] == "score"
    assert urgency["legend"] == {"0": "can wait", "1": "this week", "2": "today", "3": "right now"}
    assert list(urgency["probabilities"].keys()) == ["0", "1", "2", "3"]
    assert abs(sum(urgency["probabilities"].values()) - 1.0) < 1e-4
    assert abs(urgency["score"] - sum(i * p for i, p in enumerate(urgency["probabilities"].values()))) < 1e-4
    assert 0.0 <= urgency["confidence"] <= 1.0

    angry = answers["angry"]
    assert angry["type"] == "noul"
    assert 0.0 <= angry["noul"] <= 1.0


def test_systemone_json_state():
    global server
    server.start()
    questions = {
        "refund": {
            "type": "noul",
            "instructions": "Is a refund requested?",
            "criteria": {"false": "no refund is asked", "true": "a refund is asked"},
        },
    }
    res_obj = server.make_request("POST", "/v1/systemone", data={
        "state": {"ticket": TEST_STATE, "plan": "pro"},
        "questions": questions,
    })
    assert res_obj.status_code == 200
    # an object is given to the model as JSON text
    res_str = server.make_request("POST", "/v1/systemone", data={
        "state": '{"ticket": "' + TEST_STATE + '", "plan": "pro"}',
        "questions": questions,
    })
    assert res_str.status_code == 200
    assert res_obj.body["usage"] == res_str.body["usage"]
    assert abs(res_obj.body["answers"]["refund"]["noul"] - res_str.body["answers"]["refund"]["noul"]) < 1e-4


@pytest.mark.parametrize("data", [
    {"questions": TEST_QUESTIONS},
    {"state": TEST_STATE},
    {"state": TEST_STATE, "questions": {}},
    {"state": TEST_STATE, "questions": {"q": {"type": "unknown", "instructions": "x"}}},
    {"state": TEST_STATE, "questions": {"q": {"type": "noul"}}},
    {"state": TEST_STATE, "questions": {"q": {"type": "choice", "instructions": "x"}}},
    {"state": TEST_STATE, "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {}}}},
    {"state": TEST_STATE, "questions": {"q": {"type": "score", "instructions": "x", "criteria": ["only one"]}}},
])
def test_systemone_invalid_request(data: dict):
    global server
    server.start()
    res = server.make_request("POST", "/v1/systemone", data=data)
    assert res.status_code == 400
    assert "error" in res.body


def test_systemone_requires_embedding():
    global server
    server.server_embeddings = False
    server.start()
    res = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        "questions": TEST_QUESTIONS,
    })
    assert res.status_code == 501


# TODO: test the shared prompt prefix, it needs a small model of a type that supports it (e.g. openjev)
# it can be checked with GET /metrics: for one request, prompt_tokens_cached_total must grow by
# (shared tokens * number of child tasks) and prompt_tokens_total + prompt_tokens_cached_total == usage.input_tokens
