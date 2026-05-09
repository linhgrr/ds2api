#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


def http_json(method: str, url: str, payload=None, headers=None, timeout=60):
    data = None
    final_headers = {"Content-Type": "application/json"}
    if headers:
        final_headers.update(headers)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, headers=final_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, body, json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        parsed = None
        try:
            parsed = json.loads(body) if body else None
        except json.JSONDecodeError:
            pass
        return exc.code, body, parsed


def wait_for_ready(base_url: str, timeout_seconds: int = 30):
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            status, body, _ = http_json("GET", f"{base_url}/healthz", payload=None, headers={}, timeout=5)
            if status == 200:
                return
            last_error = f"healthz returned {status}: {body}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(1)
    raise RuntimeError(f"service did not become ready: {last_error}")


def assert_true(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def test_invalid_schema(base_url: str, bearer_token: str, model: str):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Return a JSON object."}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "BrokenSchema",
                "strict": True,
                "schema": {"type": 123},
            },
        },
    }
    status, body, parsed = http_json(
        "POST",
        f"{base_url}/v1/chat/completions",
        payload=payload,
        headers={"Authorization": f"Bearer {bearer_token}"},
    )
    assert_true(status == 400, f"expected 400 for invalid schema, got {status}: {body}")
    detail = json.dumps(parsed, ensure_ascii=False) if parsed is not None else body
    assert_true("invalid response_format json_schema" in detail, f"unexpected invalid-schema response: {detail}")
    print("[PASS] invalid schema rejected with 400")


def test_chat_structured_output(base_url: str, bearer_token: str, model: str):
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": "Return a JSON object with title='hello' and score=9."}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "ScoreCard",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["title", "score"],
                    "properties": {
                        "title": {"type": "string"},
                        "score": {"type": "integer"},
                    },
                },
            },
        },
    }
    status, body, parsed = http_json(
        "POST",
        f"{base_url}/v1/chat/completions",
        payload=payload,
        headers={"Authorization": f"Bearer {bearer_token}"},
        timeout=120,
    )
    assert_true(status == 200, f"chat structured output failed: {status} {body}")
    choice = parsed["choices"][0]["message"]
    content = choice.get("content")
    parsed_obj = choice.get("parsed")
    assert_true(isinstance(content, str) and content.strip().startswith("{"), f"unexpected chat content: {content!r}")
    decoded = json.loads(content)
    assert_true(isinstance(decoded, dict), f"chat content is not JSON object: {decoded!r}")
    assert_true(parsed_obj == decoded, f"message.parsed mismatch: parsed={parsed_obj!r} content={decoded!r}")
    assert_true(isinstance(decoded.get("score"), int), f"score is not integer: {decoded!r}")
    print("[PASS] chat structured output valid and parsed")


def test_responses_structured_output(base_url: str, bearer_token: str, model: str):
    payload = {
        "model": model,
        "stream": False,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "Return JSON with ok=true and count=2."}]}],
        "text": {
            "format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ResponseCard",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["ok", "count"],
                        "properties": {
                            "ok": {"type": "boolean"},
                            "count": {"type": "integer"},
                        },
                    },
                },
            }
        },
    }
    status, body, parsed = http_json(
        "POST",
        f"{base_url}/v1/responses",
        payload=payload,
        headers={"Authorization": f"Bearer {bearer_token}"},
        timeout=120,
    )
    assert_true(status == 200, f"responses structured output failed: {status} {body}")
    output_text = parsed.get("output_text")
    output_parsed = parsed.get("output_parsed")
    assert_true(isinstance(output_text, str) and output_text.strip().startswith("{"), f"unexpected responses output_text: {output_text!r}")
    decoded = json.loads(output_text)
    assert_true(output_parsed == decoded, f"output_parsed mismatch: parsed={output_parsed!r} content={decoded!r}")
    assert_true(isinstance(decoded.get("count"), int), f"count is not integer: {decoded!r}")
    print("[PASS] responses structured output valid and parsed")


def main():
    parser = argparse.ArgumentParser(description="E2E structured-output checks for DS2API")
    parser.add_argument("--base-url", default=os.getenv("DS2API_BASE_URL", "http://127.0.0.1:5001"))
    parser.add_argument("--model", default=os.getenv("DS2API_TEST_MODEL", "deepseek-v4-flash"))
    parser.add_argument(
        "--token",
        default=os.getenv("DS2API_DIRECT_BEARER", os.getenv("DEEPSEEK_TOKEN", "")),
        help="Bearer token. For invalid-schema-only test, any non-empty string works. For live upstream tests, provide a real DeepSeek token.",
    )
    parser.add_argument("--skip-live", action="store_true", help="Only run local invalid-schema coverage.")
    args = parser.parse_args()

    test_token = args.token or "direct-token-placeholder"
    print(f"[INFO] waiting for {args.base_url}")
    wait_for_ready(args.base_url)
    print("[INFO] service ready")

    test_invalid_schema(args.base_url, test_token, args.model)

    if args.skip_live:
        print("[INFO] skip_live enabled; success-path upstream tests skipped")
        return 0

    if not args.token:
        print("[INFO] no real token provided; success-path upstream tests skipped")
        return 0

    test_chat_structured_output(args.base_url, args.token, args.model)
    test_responses_structured_output(args.base_url, args.token, args.model)
    print("[PASS] all structured output checks completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
