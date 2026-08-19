"""Deterministic, resource-limited TACO correctness rewards."""

from __future__ import annotations

import hashlib
import json
import os
import re
import resource
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any


_FENCED_CODE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class ExecutionLimits:
    max_tests: int = 32
    timeout_seconds: float = 3.0
    memory_mb: int = 1_024


def extract_python_code(completion: str) -> str:
    blocks = _FENCED_CODE.findall(completion or "")
    return (blocks[-1] if blocks else completion or "").strip()


def parse_test_spec(raw: str | dict[str, Any]) -> dict[str, Any]:
    spec = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(spec, dict):
        raise ValueError("input_output must decode to an object.")
    inputs = spec.get("inputs")
    outputs = spec.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list) or not inputs:
        raise ValueError("input_output requires non-empty inputs and outputs lists.")
    if len(inputs) != len(outputs):
        raise ValueError("input_output inputs and outputs must have equal length.")
    return spec


def select_test_indices(count: int, problem_id: str, limit: int = 32) -> list[int]:
    if count <= limit:
        return list(range(count))
    scored = []
    for index in range(count):
        digest = hashlib.sha256(f"{problem_id}:{index}".encode()).digest()
        scored.append((digest, index))
    return sorted(index for _, index in sorted(scored)[:limit])


def _limit_process(memory_mb: int) -> None:
    memory_bytes = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def _normalize_stdout(value: Any) -> str:
    return "\n".join(line.rstrip() for line in str(value).strip().splitlines()).strip()


def _run_stdin_case(code: str, case_input: Any, expected: Any, limits: ExecutionLimits) -> bool:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONHASHSEED": "0",
        "NO_PROXY": "*",
        "http_proxy": "http://127.0.0.1:9",
        "https_proxy": "http://127.0.0.1:9",
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-S", "-c", code],
            input=str(case_input),
            text=True,
            capture_output=True,
            timeout=limits.timeout_seconds,
            env=env,
            preexec_fn=lambda: _limit_process(limits.memory_mb),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0 and _normalize_stdout(completed.stdout) == _normalize_stdout(expected)


def _function_harness(code: str, fn_name: str, case_input: Any) -> str:
    parsed_input = case_input
    if isinstance(case_input, str):
        try:
            parsed_input = json.loads(case_input)
        except json.JSONDecodeError:
            parsed_input = case_input
    encoded_input = json.dumps(parsed_input)
    return f"""{code}

import json as _json
_raw_args = _json.loads({encoded_input!r})
_target = globals().get({fn_name!r})
if _target is None:
    _solution = globals().get("Solution")
    if _solution is None:
        raise NameError("Neither function nor Solution class was defined")
    _target = getattr(_solution(), {fn_name!r})
if isinstance(_raw_args, list):
    _result = _target(*_raw_args)
else:
    _result = _target(_raw_args)
print(_json.dumps(_result, sort_keys=True, separators=(",", ":")))
"""


def _run_function_case(
    code: str,
    fn_name: str,
    case_input: Any,
    expected: Any,
    limits: ExecutionLimits,
) -> bool:
    harness = _function_harness(code, fn_name, case_input)
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-S", "-c", harness],
            text=True,
            capture_output=True,
            timeout=limits.timeout_seconds,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "NO_PROXY": "*"},
            preexec_fn=lambda: _limit_process(limits.memory_mb),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if completed.returncode != 0:
        return False
    try:
        observed = json.loads(completed.stdout.strip())
        expected_value = json.loads(expected) if isinstance(expected, str) else expected
        return observed == expected_value
    except (json.JSONDecodeError, TypeError):
        return _normalize_stdout(completed.stdout) == _normalize_stdout(expected)


def evaluate_completion(
    completion: str,
    input_output: str | dict[str, Any],
    problem_id: str,
    limits: ExecutionLimits | None = None,
) -> float:
    limits = limits or ExecutionLimits()
    code = extract_python_code(completion)
    if not code:
        return 0.0
    try:
        compile(code, "<candidate>", "exec")
        spec = parse_test_spec(input_output)
    except (SyntaxError, ValueError, json.JSONDecodeError):
        return 0.0

    indices = select_test_indices(len(spec["inputs"]), problem_id, limits.max_tests)
    fn_name = spec.get("fn_name")
    passed = 0
    for index in indices:
        if fn_name:
            ok = _run_function_case(
                code, str(fn_name), spec["inputs"][index], spec["outputs"][index], limits
            )
        else:
            ok = _run_stdin_case(
                code, spec["inputs"][index], spec["outputs"][index], limits
            )
        passed += int(ok)
    return passed / len(indices) if indices else 0.0


def reward_code_correctness(
    completions: list[str],
    input_output: list[str],
    problem_id: list[str],
    **_: Any,
) -> list[float]:
    limits = ExecutionLimits(
        max_tests=int(os.environ.get("CODE_REWARD_MAX_TESTS", "32")),
        timeout_seconds=float(os.environ.get("CODE_REWARD_TIMEOUT", "3")),
        memory_mb=int(os.environ.get("CODE_REWARD_MEMORY_MB", "1024")),
    )
    workers = max(1, int(os.environ.get("CODE_REWARD_WORKERS", "8")))
    with ThreadPoolExecutor(max_workers=min(workers, len(completions) or 1)) as pool:
        futures = [
            pool.submit(evaluate_completion, completion, tests, pid, limits)
            for completion, tests, pid in zip(
                completions, input_output, problem_id, strict=True
            )
        ]
        return [future.result() for future in futures]
