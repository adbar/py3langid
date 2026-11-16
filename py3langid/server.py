"""WSGI-compatible langid web service (`langid -s` serves it)."""

import json
from collections.abc import Callable
from http import HTTPStatus
from typing import TYPE_CHECKING
from urllib.parse import parse_qs

from .langid import classify, rank

if TYPE_CHECKING:
    from _typeshed.wsgi import StartResponse, WSGIEnvironment


def _detect(data: str | bytes) -> dict[str, object]:
    lang, conf = classify(data)
    return {"language": lang, "confidence": conf}


_ROUTES: dict[str, Callable[[str | bytes], object]] = {"detect": _detect, "rank": rank}


def application(environ: "WSGIEnvironment", start_response: "StartResponse") -> list[bytes]:
    """WSGI-compatible langid web service."""
    path = environ.get("PATH_INFO", "").strip("/").partition("/")[0]
    handler = _ROUTES.get(path)
    if handler is None:
        return _return_response(start_response, 404, None, "Not found")

    method = environ["REQUEST_METHOD"]
    if method not in ("GET", "POST", "PUT"):
        return _return_response(start_response, 405, None, f"{method} not allowed")

    data = _get_data(environ, method)
    if not data:
        detail = "Missing q parameter" if data is None else "No data provided"
        return _return_response(start_response, 400, None, detail)

    return _return_response(start_response, 200, handler(data), None)


def _get_data(environ: "WSGIEnvironment", method: str) -> bytes | None:
    """Text to classify: empty without a body or value, None without a q field."""
    if method == "GET":
        form: str = environ.get("QUERY_STRING", "")
    else:
        try:
            length = int(environ.get("CONTENT_LENGTH", 0))
        except ValueError:
            return b""
        if length <= 0:
            return b""
        data: bytes = environ["wsgi.input"].read(length)
        if method == "PUT":
            return data
        form = data.decode("latin-1")
    # WSGI strings are bytes decoded as Latin-1: the round trip restores them
    values = parse_qs(form, keep_blank_values=True, encoding="latin-1").get("q")
    return values[0].encode("latin-1") if values else None


def _return_response(
    start_response: "StartResponse",
    status_code: int,
    response_data: object,
    response_details: str | None,
) -> list[bytes]:
    status = HTTPStatus(status_code)
    response = {
        "responseData": response_data,
        "responseStatus": status_code,
        "responseDetails": response_details,
    }
    headers = [("Content-type", "application/json; charset=utf-8")]
    start_response(f"{status.value} {status.phrase}", headers)
    return [json.dumps(response).encode("utf-8")]
