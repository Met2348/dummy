"""简化的 JSON Schema 校验器（纯 stdlib，不依赖第三方 jsonschema 库）。

用于验证一个 function-calling 式工具调用的参数字典是否符合给定 schema：
- 必填字段检查（required）
- 基础类型检查（string/integer/boolean/number/array/object）
并附一个"失败重试"模拟：校验失败时生成明确的错误反馈（哪个字段缺失/类型不对），
再用预先构造好的"修正后"参数模拟一次重试。
"""
from __future__ import annotations

TYPE_MAP = {
    "string": str,
    "integer": int,
    "boolean": bool,
    "number": (int, float),
    "array": list,
    "object": dict,
}


def validate(schema: dict, instance: dict) -> tuple[bool, list[str]]:
    """校验 instance 是否符合 schema。返回 (是否通过, 错误信息列表)。

    schema 形如：
        {"type": "object",
         "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
         "required": ["city"]}
    """
    errors: list[str] = []
    required = schema.get("required", [])
    properties = schema.get("properties", {})

    for field in required:
        if field not in instance:
            errors.append(f"missing required field: '{field}'")

    for field, value in instance.items():
        if field not in properties:
            continue  # 简化实现：未声明的额外字段不做严格模式校验，直接忽略
        expected_type = properties[field].get("type")
        py_type = TYPE_MAP.get(expected_type)
        if py_type is None:
            continue
        # bool 是 int 的子类，Python 里 isinstance(True, int) 为 True，
        # 必须单独排除，否则 True/False 会被误判成合法的 integer。
        if expected_type == "integer" and isinstance(value, bool):
            errors.append(f"field '{field}' expected type integer, got bool")
            continue
        if not isinstance(value, py_type):
            errors.append(
                f"field '{field}' expected type {expected_type}, got {type(value).__name__}"
            )

    return (len(errors) == 0, errors)


def call_with_retry(
    schema: dict, instance: dict, corrected_instance: dict, max_retries: int = 1,
) -> list[dict]:
    """模拟一次 function-calling 的失败重试流程：
    先校验 instance，若失败，生成结构化错误反馈，再用预先构造好的
    corrected_instance 重试一次（最多 max_retries 次）。返回每次尝试的记录。
    """
    attempts: list[dict] = []
    ok, errors = validate(schema, instance)
    attempts.append({"instance": instance, "ok": ok, "errors": errors})
    if ok or max_retries <= 0:
        return attempts

    ok2, errors2 = validate(schema, corrected_instance)
    attempts.append({"instance": corrected_instance, "ok": ok2, "errors": errors2})
    return attempts


# 示例 schema：一个 get_weather(city, date, unit) 工具
WEATHER_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "date": {"type": "string"},
        "unit": {"type": "string"},
        "days": {"type": "integer"},
    },
    "required": ["city", "date"],
}


def _self_test() -> None:
    # 合法调用：应通过校验
    ok, errors = validate(WEATHER_SCHEMA, {"city": "Beijing", "date": "2026-07-12"})
    assert ok is True
    assert errors == []

    # 缺字段：应被拒绝，且错误信息里准确指出缺的是 date
    ok, errors = validate(WEATHER_SCHEMA, {"city": "Beijing"})
    assert ok is False
    assert any("date" in e for e in errors)

    # 类型错：date 传成 int，应被拒绝且报出 date 字段、期望类型 string
    ok, errors = validate(WEATHER_SCHEMA, {"city": "Beijing", "date": 20260712})
    assert ok is False
    assert any("date" in e and "expected type string" in e for e in errors)

    # bool 不应被误判为合法 integer
    ok, errors = validate(
        WEATHER_SCHEMA, {"city": "Beijing", "date": "2026-07-12", "days": True},
    )
    assert ok is False
    assert any("days" in e and "bool" in e for e in errors)

    # 合法的 integer 字段应通过
    ok, errors = validate(
        WEATHER_SCHEMA, {"city": "Beijing", "date": "2026-07-12", "days": 3},
    )
    assert ok is True

    # 重试流程：先错（缺 date）后对（补上 date）
    bad_call = {"city": "Beijing"}
    fixed_call = {"city": "Beijing", "date": "2026-07-12"}
    attempts = call_with_retry(WEATHER_SCHEMA, bad_call, fixed_call)
    assert len(attempts) == 2
    assert attempts[0]["ok"] is False
    assert any("date" in e for e in attempts[0]["errors"])
    assert attempts[1]["ok"] is True

    # 一次性合法调用不应触发重试
    attempts_ok = call_with_retry(WEATHER_SCHEMA, fixed_call, fixed_call)
    assert len(attempts_ok) == 1
    assert attempts_ok[0]["ok"] is True

    print("[PASS] tool_calling_schema: 必填/类型校验 + 结构化错误反馈 + 修正重试")


if __name__ == "__main__":
    _self_test()
