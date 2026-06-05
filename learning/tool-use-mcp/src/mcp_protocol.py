"""MCP JSON-RPC 2.0 envelope helpers."""
from __future__ import annotations


JSONRPC_VERSION = "2.0"
PROTOCOL_VERSION = "2024-11-05"


ERROR_CODES = {
    "parse_error": -32700,
    "invalid_request": -32600,
    "method_not_found": -32601,
    "invalid_params": -32602,
    "internal_error": -32603,
}


def make_request(req_id: int, method: str, params: dict | None = None) -> dict:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": req_id,
        "method": method,
        "params": params or {},
    }


def make_response(req_id: int, result: dict) -> dict:
    return {"jsonrpc": JSONRPC_VERSION, "id": req_id, "result": result}


def make_error(req_id: int, code: int, message: str, data: dict | None = None) -> dict:
    err: dict = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": JSONRPC_VERSION, "id": req_id, "error": err}


def is_response(msg: dict) -> bool:
    return "result" in msg


def is_error(msg: dict) -> bool:
    return "error" in msg


def _self_test() -> None:
    req = make_request(1, "tools/list")
    assert req["jsonrpc"] == "2.0"
    assert req["method"] == "tools/list"

    resp = make_response(1, {"tools": []})
    assert is_response(resp) and not is_error(resp)

    err = make_error(1, ERROR_CODES["method_not_found"], "nope")
    assert is_error(err) and err["error"]["code"] == -32601
    print("[OK] mcp_protocol._self_test passed")


if __name__ == "__main__":
    _self_test()
