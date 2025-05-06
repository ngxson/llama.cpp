import pytest
from utils import *
import base64
import requests

server: ServerProcess

IMG_URL_0 = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/11_truck.png"
IMG_URL_1 = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/91_cat.png"

response = requests.get(IMG_URL_0)
response.raise_for_status() # Raise an exception for bad status codes
IMG_BASE64_0 = "data:image/png;base64," + base64.b64encode(response.content).decode("utf-8")


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinygemma3()


@pytest.mark.parametrize(
    "image_url, success, re_content",
    [
        # test model is trained on CIFAR-10, but it's quite dumb due to small size
        (IMG_URL_0,                True, "(cat)+"),
        (IMG_BASE64_0,             True, "(cat)+"),
        (IMG_URL_1,                True, "(frog)+"),
        ("malformed",              False, None),
        ("https://google.com/404", False, None), # non-existent image
        ("https://ggml.ai",        False, None), # non-image data
    ]
)
def test_vision_chat_completion(image_url, success, re_content):
    global server
    server.start(timeout_seconds=60) # vision model may take longer to load due to download size
    res = server.make_request("POST", "/chat/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "What is this:\n"},
                {"type": "image_url", "image_url": {
                    "url": image_url,
                }},
            ]},
        ],
    })
    if success:
        assert res.status_code == 200
        choice = res.body["choices"][0]
        assert "assistant" == choice["message"]["role"]
        assert match_regex(re_content, choice["message"]["content"])
    else:
        assert res.status_code != 200

