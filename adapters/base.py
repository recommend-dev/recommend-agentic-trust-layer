"""Shared HTTP session + the uniform item shape search lanes return."""
from dataclasses import dataclass

import requests

_session = None


def session():
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers["User-Agent"] = "confidence-check/0.1"
    return _session


@dataclass
class RawItem:
    url: str = ""
    title: str = ""
    content: str = ""
    summary: str = ""
    imageurl: str = ""
    postdate: str = ""
    author: str = ""
    hash: str = ""
