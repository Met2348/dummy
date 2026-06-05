"""Pydantic AI-style typed agent mock — schema-validated output with retry."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
import re


@dataclass
class TypedField:
    name: str
    type_: type
    required: bool = True


class TypedSchema:
    """Mini BaseModel — declares fields and validates dicts."""
    fields: list[TypedField] = []

    @classmethod
    def validate(cls, data: dict) -> "TypedSchema":
        instance = cls()
        for f in cls.fields:
            if f.required and f.name not in data:
                raise ValueError(f"missing required: {f.name}")
            value = data.get(f.name)
            if value is not None and not isinstance(value, f.type_):
                try:
                    coerced = f.type_(value)
                    setattr(instance, f.name, coerced)
                except Exception as e:
                    raise ValueError(f"{f.name}: cannot coerce to {f.type_.__name__}: {e}") from e
            else:
                setattr(instance, f.name, value)
        return instance


class TypedAgent:
    def __init__(
        self,
        model: str,
        result_type: type,
        system_prompt: str = "",
        max_retries: int = 3,
    ):
        self.model = model
        self.result_type = result_type
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.tools: dict[str, Callable] = {}

    def tool(self, fn: Callable) -> Callable:
        self.tools[fn.__name__] = fn
        return fn

    def run_sync(self, prompt: str, llm_fn: Callable[[str], dict]) -> Any:
        last_error = None
        for attempt in range(self.max_retries):
            raw = llm_fn(prompt + (f"\n(retry {attempt}, prev error: {last_error})" if last_error else ""))
            try:
                validated = self.result_type.validate(raw)
                return validated
            except ValueError as e:
                last_error = str(e)
        raise RuntimeError(f"agent failed after {self.max_retries} retries: {last_error}")


class Weather(TypedSchema):
    fields = [
        TypedField("temperature", float, required=True),
        TypedField("conditions", str, required=True),
        TypedField("humidity", int, required=False),
    ]


def _self_test() -> None:
    agent = TypedAgent("mock-llm", result_type=Weather, system_prompt="weather expert", max_retries=3)

    @agent.tool
    def get_weather(city: str) -> dict:
        return {"temp": 22}

    def good_llm(prompt: str) -> dict:
        return {"temperature": 22.5, "conditions": "sunny", "humidity": 65}

    result = agent.run_sync("Tokyo weather", good_llm)
    assert result.temperature == 22.5
    assert result.conditions == "sunny"
    assert result.humidity == 65

    attempts = {"n": 0}
    def flaky_llm(prompt: str) -> dict:
        attempts["n"] += 1
        if attempts["n"] < 2:
            return {"temperature": "twenty-two", "conditions": "sunny"}
        return {"temperature": 22.0, "conditions": "sunny"}
    result2 = agent.run_sync("Tokyo", flaky_llm)
    assert result2.temperature == 22.0
    assert attempts["n"] == 2

    def bad_llm(prompt: str) -> dict:
        return {"temperature": "X"}
    try:
        agent.run_sync("Tokyo", bad_llm)
        assert False, "should have raised"
    except RuntimeError as e:
        assert "retries" in str(e)
    print("[OK] pydantic_ai_style._self_test passed")


if __name__ == "__main__":
    _self_test()
