import json
from io import BytesIO
from unittest.mock import MagicMock
from urllib.parse import quote_plus

import pytest

from py3langid.server import application


def _request(path, method="GET", body=None, query=None, content_length="auto"):
    """Run a WSGI request and return (status, payload, headers)."""
    start_response = MagicMock()
    environ = {"REQUEST_METHOD": method, "PATH_INFO": path}
    if query is not None:
        environ["QUERY_STRING"] = query
    if body is not None:
        environ["wsgi.input"] = BytesIO(body)
        if content_length == "auto":
            environ["CONTENT_LENGTH"] = str(len(body))
        elif content_length is not None:
            environ["CONTENT_LENGTH"] = content_length
    response = application(environ, start_response)
    status = start_response.call_args[0][0]
    headers = dict(start_response.call_args[0][1])
    payload = json.loads(response[0].decode("utf-8"))
    return status, payload, headers


@pytest.mark.parametrize("path", ["/detect", "/rank"])
@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("GET", {"query": "q=This+is+a+test"}),
        ("POST", {"body": b"q=This+is+a+test"}),
        ("PUT", {"body": b"This is a test"}),
    ],
)
def test_ok(path, method, kwargs):
    status, payload, _ = _request(path, method, **kwargs)
    assert status == "200 OK"
    data = payload["responseData"]
    if path == "/detect":
        assert data["language"] == "en"
    else:
        assert data[0][0] == "en"


def test_invalid_method():
    status, _, _ = _request("/detect", "DELETE")
    assert status == "405 Method Not Allowed"


@pytest.mark.parametrize("path", ["/invalid", ""])
def test_invalid_path(path):
    status, _, _ = _request(path)
    assert status == "404 Not Found"


RU = "Это русский текст для проверки"


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("GET", {"query": "q=" + quote_plus(RU)}),
        ("GET", {"query": "q=" + RU.encode().decode("latin-1")}),  # raw bytes, as WSGI passes them
        ("POST", {"body": ("q=" + quote_plus(RU)).encode()}),
        ("POST", {"body": b"q=" + RU.encode()}),
    ],
)
def test_non_ascii(method, kwargs):
    """percent-encoded or raw UTF-8 (parse_qs on bytes is ASCII-only on Python 3.10)"""
    _, payload, _ = _request("/detect", method, **kwargs)
    assert payload["responseData"]["language"] == "ru"


def test_no_query_string():
    status, payload, _ = _request("/detect")
    assert status == "400 Bad Request"
    assert payload["responseData"] is None
    assert payload["responseDetails"] == "Missing q parameter"


def test_get_missing_q_param():
    status, _, _ = _request("/detect", query="x=hello")
    assert status == "400 Bad Request"


@pytest.mark.parametrize("body", [b"x=hello", b"This is a test", b'{"q": "This is a test"}'])
def test_post_missing_q_param(body):
    status, payload, _ = _request("/detect", "POST", body=body)
    assert status == "400 Bad Request"
    assert payload["responseDetails"] == "Missing q parameter"


@pytest.mark.parametrize("method,kwargs", [("GET", {"query": "q="}), ("POST", {"body": b"q="})])
def test_empty_q_param(method, kwargs):
    status, payload, _ = _request("/detect", method, **kwargs)
    assert status == "400 Bad Request"
    assert payload["responseDetails"] == "No data provided"


def test_content_type():
    _, _, headers = _request("/detect", query="q=test")
    assert headers["Content-type"] == "application/json; charset=utf-8"


def test_missing_content_length():
    status, payload, _ = _request("/detect", "POST", body=b"q=test", content_length=None)
    assert status == "400 Bad Request"
    assert payload["responseDetails"] == "No data provided"


def test_invalid_content_length():
    status, _, _ = _request("/detect", "POST", body=b"q=test", content_length="abc")
    assert status == "400 Bad Request"
