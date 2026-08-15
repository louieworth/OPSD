"""Teacher-refinement prompts and the fixed vLLM client used by TRD.

The refinement model is deliberately generation-only.  It never joins a
distributed communicator and its weights are never synchronized with the
student.  The token IDs returned by the server are the canonical IDs for the
teacher-side KL prefix; callers should not decode and tokenize them again.
"""

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover - exercised only in an incomplete environment
    requests = None


MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

# These are the math refinement prompts released with TRD.  Keep their
# whitespace and marker names aligned with the reference implementation: the
# markers are also the boundaries used by component-aware truncation below.
PROMPT_TEMPLATE_OPSD_REFINE = """
Your task is to rewrite your mathematical solution using the reference solution as guidance.

**Problem:**
{PROBLEM}

**Reference Solution:**
{EXPERT_SOLUTION}

**Your Initial Solution:**
{INITIAL_RESPONSE}

**Instructions:**
1. Review the reference solution to understand the target reasoning and method
2. Rewrite your solution so it is consistent with the reference solution
3. Keep useful parts of your original structure and style when appropriate
4. Output ONLY the rewritten solution
"""

PROMPT_TEMPLATE_OPD_REFINE = """
Your task is to rewrite your mathematical solution.

**Problem:**
{PROBLEM}

**Your Initial Solution:**
{INITIAL_RESPONSE}

**Instructions:**
1. Preserve the overall structure and reasoning path of your original solution
2. Identify and fix errors in computation or logic
3. Keep correct intermediate steps and meaningful work
4. Output ONLY the rewritten solution
"""

_INITIAL_TRUNCATION_NOTICE = "\n[... initial solution truncated ...]\n"
_REFERENCE_TRUNCATION_NOTICE = "\n[... reference solution truncated ...]\n"


class RefinementPromptError(ValueError):
    """Raised when a valid refinement prompt cannot fit in the context."""


class RefinementServerError(RuntimeError):
    """Raised when the fixed teacher server violates its HTTP contract."""


@dataclass(frozen=True)
class RefinementGenerationBatch:
    """Canonical tokenized prompts and samples returned by the teacher server."""

    prompt_ids: list[list[int]]
    completion_ids: list[list[int]]
    logprobs: list[list[float]]


def _validate_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    """Encode rendered prompt text without injecting another set of special tokens."""
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except AttributeError as exc:
        raise TypeError("The refinement tokenizer must provide encode().") from exc
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if not isinstance(encoded, list) or any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in encoded
    ):
        raise TypeError("tokenizer.encode() must return a flat list of non-negative token IDs.")
    return encoded


def _decode_prefix(tokenizer: Any, token_ids: list[int]) -> str:
    try:
        text = tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        # Some lightweight/custom tokenizers do not expose the cleanup option.
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
    except AttributeError as exc:
        raise TypeError("The refinement tokenizer must provide decode().") from exc
    if not isinstance(text, str):
        raise TypeError("tokenizer.decode() must return a string.")
    return text


def _refinement_content(
    alg: str,
    problem: str,
    initial_response: str,
    reference_solution: str | None,
) -> str:
    if alg == "opd":
        prompt = (
            PROMPT_TEMPLATE_OPD_REFINE.replace("{PROBLEM}", problem).replace(
                "{INITIAL_RESPONSE}", initial_response
            )
        )
    else:
        # ``reference_solution is None`` is rejected by the public builder.
        prompt = (
            PROMPT_TEMPLATE_OPSD_REFINE.replace("{PROBLEM}", problem)
            .replace("{EXPERT_SOLUTION}", reference_solution or "")
            .replace("{INITIAL_RESPONSE}", initial_response)
        )
    return prompt + " " + MATH_INSTRUCTION


def _render_refinement_prompt(
    tokenizer: Any,
    alg: str,
    problem: str,
    initial_response: str,
    reference_solution: str | None,
    teacher_thinking: bool,
) -> str:
    content = _refinement_content(alg, problem, initial_response, reference_solution)
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=teacher_thinking,
        )
    except AttributeError as exc:
        raise TypeError("The refinement tokenizer must provide apply_chat_template().") from exc
    if not isinstance(rendered, str):
        raise TypeError("tokenizer.apply_chat_template(..., tokenize=False) must return a string.")
    return rendered


def _truncate_component_to_fit(
    *,
    tokenizer: Any,
    component: str,
    notice: str,
    prompt_budget: int,
    render: Any,
) -> tuple[str, str]:
    """Keep the longest token-prefix of one component that fits, if possible.

    TRD truncates the tail of the initial solution before touching the
    reference solution.  Decoding a token prefix mirrors that behavior without
    slicing Unicode strings or potentially splitting a tokenizer token.
    """
    component_ids = _token_ids(tokenizer, component)
    if not component_ids:
        rendered = render(component)
        return component, rendered

    def candidate(keep_tokens: int) -> tuple[str, str]:
        shortened = _decode_prefix(tokenizer, component_ids[:keep_tokens]) + notice
        return shortened, render(shortened)

    # If removing the whole block is not enough, lock it at its smallest
    # official representation and let the next component be truncated.
    shortest_component, shortest_rendered = candidate(0)
    if len(_token_ids(tokenizer, shortest_rendered)) > prompt_budget:
        return shortest_component, shortest_rendered

    # The unmodified form was already found to overflow.  Binary search the
    # largest retained prefix below the full component length.
    low, high = 0, len(component_ids) - 1
    best_component, best_rendered = shortest_component, shortest_rendered
    while low <= high:
        midpoint = (low + high) // 2
        shortened, rendered = candidate(midpoint)
        if len(_token_ids(tokenizer, rendered)) <= prompt_budget:
            best_component, best_rendered = shortened, rendered
            low = midpoint + 1
        else:
            high = midpoint - 1
    return best_component, best_rendered


def build_refinement_prompt(
    tokenizer: Any,
    *,
    alg: str,
    problem: str,
    initial_response: str,
    reference_solution: str | None = None,
    teacher_thinking: bool = True,
    max_model_len: int = 20_000,
    max_refinement_length: int = 1_024,
) -> str:
    """Build and render one official TRD math teacher-refinement prompt.

    The initial solution (``y_o``) is truncated from its tail first.  For OPSD,
    the reference solution (``y*``) is truncated from its tail only if removing
    ``y_o`` is insufficient.  The problem, rewrite instructions, chat-template
    framing, and the full requested output budget are never silently removed.

    Returns:
        The raw rendered chat text to send to the fixed teacher vLLM server.
    """
    if alg not in {"opd", "opsd"}:
        raise ValueError(f"Unsupported refinement algorithm: {alg!r}; expected 'opd' or 'opsd'.")
    if not isinstance(problem, str) or not isinstance(initial_response, str):
        raise TypeError("problem and initial_response must be strings.")
    if alg == "opsd" and reference_solution is None:
        raise ValueError("OPSD teacher refinement requires reference_solution (y*).")
    if reference_solution is not None and not isinstance(reference_solution, str):
        raise TypeError("reference_solution must be a string or None.")
    if not isinstance(teacher_thinking, bool):
        raise TypeError("teacher_thinking must be a boolean.")
    _validate_positive_int(max_model_len, "max_model_len")
    _validate_positive_int(max_refinement_length, "max_refinement_length")
    if max_refinement_length >= max_model_len:
        raise RefinementPromptError(
            "max_refinement_length must leave at least one token for the rendered prompt; "
            f"got {max_refinement_length=} and {max_model_len=}."
        )

    prompt_budget = max_model_len - max_refinement_length
    current_initial = initial_response
    current_reference = reference_solution
    rendered = _render_refinement_prompt(
        tokenizer,
        alg,
        problem,
        current_initial,
        current_reference,
        teacher_thinking,
    )
    if len(_token_ids(tokenizer, rendered)) <= prompt_budget:
        return rendered

    def render_with_initial(value: str) -> str:
        return _render_refinement_prompt(
            tokenizer,
            alg,
            problem,
            value,
            current_reference,
            teacher_thinking,
        )

    current_initial, rendered = _truncate_component_to_fit(
        tokenizer=tokenizer,
        component=current_initial,
        notice=_INITIAL_TRUNCATION_NOTICE,
        prompt_budget=prompt_budget,
        render=render_with_initial,
    )
    if len(_token_ids(tokenizer, rendered)) <= prompt_budget:
        return rendered

    if alg == "opsd":

        def render_with_reference(value: str) -> str:
            return _render_refinement_prompt(
                tokenizer,
                alg,
                problem,
                current_initial,
                value,
                teacher_thinking,
            )

        current_reference, rendered = _truncate_component_to_fit(
            tokenizer=tokenizer,
            component=current_reference or "",
            notice=_REFERENCE_TRUNCATION_NOTICE,
            prompt_budget=prompt_budget,
            render=render_with_reference,
        )
        if len(_token_ids(tokenizer, rendered)) <= prompt_budget:
            return rendered

    minimum_prompt_length = len(_token_ids(tokenizer, rendered))
    raise RefinementPromptError(
        "The TRD refinement prompt cannot fit without truncating the problem or instructions: "
        f"minimum rendered prompt is {minimum_prompt_length} tokens but only {prompt_budget} tokens remain after "
        f"reserving {max_refinement_length} refinement tokens."
    )


def _validate_finite_number(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite real number; got {value!r}.")
    return float(value)


class TeacherVLLMClient:
    """HTTP-only client for a fixed, generation-only ``trl vllm-serve``.

    Construction performs one health request and one world-size request.  No
    retry loop is used: lifecycle/readiness belongs to the launcher, and any
    request failure is surfaced immediately to every training rank by the
    caller's distributed error handling.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int | None = None,
        *,
        server_port: int | None = None,
        connect_timeout: float = 10.0,
        read_timeout: float = 1_800.0,
        expected_world_size: int | None = None,
        max_model_len: int = 20_000,
        session: Any | None = None,
    ) -> None:
        if requests is None and session is None:
            raise ImportError("TeacherVLLMClient requires the 'requests' package.")
        if not isinstance(host, str) or not host or "://" in host:
            raise ValueError("host must be a non-empty hostname or IP address without a URL scheme.")
        if port is None:
            port = 8002 if server_port is None else server_port
        elif server_port is not None and server_port != port:
            raise ValueError(f"Conflicting port values: {port=} and {server_port=}.")
        _validate_positive_int(port, "port")
        if port > 65_535:
            raise ValueError(f"port must be <= 65535; got {port}.")
        self.connect_timeout = _validate_finite_number(connect_timeout, "connect_timeout")
        self.read_timeout = _validate_finite_number(read_timeout, "read_timeout")
        if self.connect_timeout <= 0 or self.read_timeout <= 0:
            raise ValueError("connect_timeout and read_timeout must both be greater than zero.")
        if expected_world_size is not None:
            _validate_positive_int(expected_world_size, "expected_world_size")
        _validate_positive_int(max_model_len, "max_model_len")

        self.base_url = f"http://{host}:{port}"
        self.expected_world_size = expected_world_size
        self.max_model_len = max_model_len
        self._session = session if session is not None else requests.Session()
        self.world_size = self.check_server()

    @property
    def timeout(self) -> tuple[float, float]:
        """The ``requests`` (connect timeout, read timeout) pair."""
        return self.connect_timeout, self.read_timeout

    def _request_json(self, method: str, path: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            if method == "GET":
                response = self._session.get(url, timeout=self.timeout)
            elif method == "POST":
                response = self._session.post(url, json=payload, timeout=self.timeout)
            else:  # pragma: no cover - all call sites are static
                raise AssertionError(f"Unsupported HTTP method: {method}")
        except Exception as exc:
            raise RefinementServerError(f"Fixed teacher request failed: {method} {url}: {exc}") from exc

        status_code = getattr(response, "status_code", None)
        if status_code != 200:
            response_text = str(getattr(response, "text", ""))[:500]
            raise RefinementServerError(
                f"Fixed teacher request failed: {method} {url} returned HTTP {status_code}: {response_text}"
            )
        try:
            body = response.json()
        except Exception as exc:
            raise RefinementServerError(f"Fixed teacher returned invalid JSON for {method} {url}.") from exc
        if not isinstance(body, dict):
            raise RefinementServerError(
                f"Fixed teacher returned {type(body).__name__}, expected a JSON object for {method} {url}."
            )
        return body

    def check_server(self) -> int:
        """Validate health and return the fixed server's TP×DP world size."""
        health = self._request_json("GET", "/health/")
        if health.get("status") != "ok":
            raise RefinementServerError(
                f"Fixed teacher health response must contain status='ok'; got {health!r}."
            )

        world_size_response = self._request_json("GET", "/get_world_size/")
        world_size = world_size_response.get("world_size")
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
            raise RefinementServerError(
                "Fixed teacher world-size response must contain a positive integer; "
                f"got {world_size_response!r}."
            )
        if self.expected_world_size is not None and world_size != self.expected_world_size:
            raise RefinementServerError(
                f"Fixed teacher world size is {world_size}, expected {self.expected_world_size}."
            )
        return world_size

    @staticmethod
    def _validate_token_matrix(
        value: Any,
        *,
        name: str,
        expected_rows: int,
        allow_empty_rows: bool,
    ) -> list[list[int]]:
        if not isinstance(value, list) or len(value) != expected_rows:
            actual_rows = len(value) if isinstance(value, list) else type(value).__name__
            raise RefinementServerError(
                f"Fixed teacher {name} must contain {expected_rows} rows; got {actual_rows}."
            )
        validated = []
        for row_index, row in enumerate(value):
            if not isinstance(row, list):
                raise RefinementServerError(f"Fixed teacher {name}[{row_index}] must be a list.")
            if not row and not allow_empty_rows:
                raise RefinementServerError(f"Fixed teacher {name}[{row_index}] is empty.")
            if any(
                isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
                for token_id in row
            ):
                raise RefinementServerError(
                    f"Fixed teacher {name}[{row_index}] must contain only non-negative integer token IDs."
                )
            validated.append(list(row))
        return validated

    def generate(
        self,
        prompts: list[str],
        *,
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 1_024,
        presence_penalty: float = 0.0,
    ) -> RefinementGenerationBatch:
        """Generate rewrites and return the server's canonical token IDs."""
        if not isinstance(prompts, list) or not prompts or any(
            not isinstance(prompt, str) or not prompt for prompt in prompts
        ):
            raise ValueError("prompts must be a non-empty list of non-empty rendered strings.")
        _validate_positive_int(n, "n")
        _validate_positive_int(max_tokens, "max_tokens")
        if max_tokens > self.max_model_len:
            raise ValueError(
                f"max_tokens ({max_tokens}) cannot exceed max_model_len ({self.max_model_len})."
            )
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k == 0 or top_k < -1:
            raise ValueError("top_k must be -1 (disabled) or a positive integer.")
        repetition_penalty = _validate_finite_number(repetition_penalty, "repetition_penalty")
        temperature = _validate_finite_number(temperature, "temperature")
        top_p = _validate_finite_number(top_p, "top_p")
        min_p = _validate_finite_number(min_p, "min_p")
        presence_penalty = _validate_finite_number(presence_penalty, "presence_penalty")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be greater than zero.")
        if temperature < 0:
            raise ValueError("temperature must be non-negative.")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1].")
        if not 0 <= min_p <= 1:
            raise ValueError("min_p must be in [0, 1].")

        payload = {
            "prompts": list(prompts),
            "images": None,
            "n": n,
            "repetition_penalty": repetition_penalty,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_tokens": max_tokens,
            # Component-aware truncation must happen before this request.  A
            # generic vLLM left-truncation would discard the problem/instructions.
            "truncate_prompt_tokens": None,
            "guided_decoding_regex": None,
            # TRL 0.26's HTTP schema forwards extra SamplingParams only via
            # generation_kwargs; presence_penalty is not a top-level field.
            "generation_kwargs": {"presence_penalty": presence_penalty},
        }
        response = self._request_json("POST", "/generate/", payload=payload)

        prompt_ids = self._validate_token_matrix(
            response.get("prompt_ids"),
            name="prompt_ids",
            expected_rows=len(prompts),
            allow_empty_rows=False,
        )
        expected_completions = len(prompts) * n
        completion_ids = self._validate_token_matrix(
            response.get("completion_ids"),
            name="completion_ids",
            expected_rows=expected_completions,
            allow_empty_rows=False,
        )
        for completion_index, completion in enumerate(completion_ids):
            if len(completion) > max_tokens:
                raise RefinementServerError(
                    f"Fixed teacher completion_ids[{completion_index}] has {len(completion)} tokens, "
                    f"exceeding max_tokens={max_tokens}."
                )
            prompt_index = completion_index // n
            total_length = len(prompt_ids[prompt_index]) + len(completion)
            if total_length > self.max_model_len:
                raise RefinementServerError(
                    f"Fixed teacher prompt/completion pair {completion_index} has {total_length} tokens, "
                    f"exceeding max_model_len={self.max_model_len}."
                )

        raw_logprobs = response.get("logprobs")
        if not isinstance(raw_logprobs, list) or len(raw_logprobs) != expected_completions:
            actual_rows = len(raw_logprobs) if isinstance(raw_logprobs, list) else type(raw_logprobs).__name__
            raise RefinementServerError(
                f"Fixed teacher logprobs must contain {expected_completions} rows; got {actual_rows}."
            )
        logprobs: list[list[float]] = []
        for row_index, (row, completion) in enumerate(zip(raw_logprobs, completion_ids, strict=True)):
            if not isinstance(row, list) or len(row) != len(completion):
                row_length = len(row) if isinstance(row, list) else type(row).__name__
                raise RefinementServerError(
                    f"Fixed teacher logprobs[{row_index}] has length {row_length}; "
                    f"expected {len(completion)}."
                )
            if any(
                isinstance(logprob, bool)
                or not isinstance(logprob, Real)
                or not math.isfinite(float(logprob))
                for logprob in row
            ):
                raise RefinementServerError(
                    f"Fixed teacher logprobs[{row_index}] must contain only finite real numbers."
                )
            logprobs.append([float(logprob) for logprob in row])

        return RefinementGenerationBatch(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            logprobs=logprobs,
        )

    def close(self) -> None:
        close = getattr(self._session, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> "TeacherVLLMClient":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
