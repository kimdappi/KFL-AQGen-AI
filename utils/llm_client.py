# utils/llm_client.py
"""
Robust LLM client wrapper using openai.OpenAI (v1+).
Features:
- Loads OPENAI_API_KEY from environment (raises if missing)
- Uses OpenAI(api_key=...) client and client.chat.completions.create(...)
- Retries with exponential backoff and jitter on transient failures
- Model fallback sequence supported via call_chat_model_json's preferred_models
- Attempts to parse JSON from model text; if parsing fails, runs a repair loop asking the model to output ONLY valid JSON
- Safe/compatible import of openai exception classes (works across openai package versions)
"""
import os
import time
import json
import re
import importlib
from typing import List, Dict, Any, Optional

# --- Config / defaults ---
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 파일 또는 환경변수에 키를 넣어 주세요.")

DEFAULT_PARAMS = {
    "temperature": 0.0,
    "top_p": 0.9,
    "max_tokens": 1024
}

# --- import OpenAI client (v1) ---
try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("openai 패키지의 OpenAI 클라이언트를 불러오지 못했습니다. 'pip install openai' 등으로 설치해 주세요.") from e

# create module-level client
_client = OpenAI(api_key=OPENAI_KEY)

# --- safe import of exception classes across openai versions ---
try:
    # older/newer distributions might expose openai.error
    from openai.error import RateLimitError, APIError, ServiceUnavailableError, Timeout
except Exception:
    # try internal _exceptions if present, else fallback to generic Exception
    try:
        _err_mod = importlib.import_module("openai._exceptions")
    except Exception:
        _err_mod = None

    def _get_exc(name: str):
        if _err_mod and hasattr(_err_mod, name):
            return getattr(_err_mod, name)
        return Exception

    RateLimitError = _get_exc("RateLimitError")
    APIError = _get_exc("APIError")
    ServiceUnavailableError = _get_exc("ServiceUnavailableError")
    Timeout = _get_exc("Timeout")

# --- utilities for JSON extraction / cleaning ---
_JSON_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def extract_json_substring(text: str) -> Optional[str]:
    """
    Try to find a JSON substring inside a larger text.
    Returns the JSON substring or None.
    """
    if not text:
        return None
    # quick regex attempt
    m = _JSON_RE.search(text)
    if m:
        return m.group(1)
    # fallback: manual bracket matching (best-effort)
    for start_char in ('{', '['):
        try:
            start = text.index(start_char)
        except ValueError:
            continue
        stack = []
        for i in range(start, len(text)):
            ch = text[i]
            if ch == start_char:
                stack.append(ch)
            elif ch == '}' and start_char == '{':
                if stack:
                    stack.pop()
                if not stack:
                    return text[start:i+1]
            elif ch == ']' and start_char == '[':
                if stack:
                    stack.pop()
                if not stack:
                    return text[start:i+1]
    return None


def _heuristic_clean_json_string(s: str) -> str:
    """
    Minor heuristics to fix some common JSON issues:
    - remove trailing commas before } or ]
    """
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def _try_parse_json_from_text(text: str) -> Optional[Any]:
    """
    Attempt to parse JSON from the given text:
    - try parse whole text
    - extract JSON-like substring and parse
    - heuristically clean and parse
    Returns parsed object or None.
    """
    if not text:
        return None
    # try raw text
    try:
        return json.loads(text)
    except Exception:
        pass
    # try substring extraction
    jstr = extract_json_substring(text)
    if not jstr:
        return None
    try:
        return json.loads(jstr)
    except Exception:
        # try heuristic cleanup
        try:
            cleaned = _heuristic_clean_json_string(jstr)
            return json.loads(cleaned)
        except Exception:
            return None


# --- core: call chat model and return text (uses OpenAI v1 client) ---
def call_chat_model_text(
    model: str,
    messages: List[Dict[str, str]],
    retries: int = 4,
    backoff_base: float = 1.0,
    timeout_seconds: Optional[float] = None,
    **kwargs
) -> str:
    """
    Call the model and return assistant text.
    Uses exponential backoff + jitter on exceptions.
    kwargs passed to client.chat.completions.create (e.g. temperature, max_tokens).
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            # call via v1 client
            # note: client.chat.completions.create expects messages as list of dicts with role/content
            resp = _client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            # Try attribute-style access (object), then dict fallback
            try:
                text = resp.choices[0].message.content
            except Exception:
                try:
                    text = resp["choices"][0]["message"]["content"]
                except Exception:
                    text = str(resp)
            return text
        except (RateLimitError, ServiceUnavailableError, Timeout, APIError) as e:
            last_exc = e
            wait = backoff_base * (2 ** (attempt - 1))
            # jitter
            jitter = (0.9 + 0.2 * (os.urandom(1)[0] / 255.0))
            wait = wait * jitter
            print(f"[LLM] transient error on model={model}, attempt={attempt}/{retries}: {e}. retry after {wait:.1f}s")
            time.sleep(wait)
        except Exception as e:
            last_exc = e
            # non-transient fallback with small sleep
            print(f"[LLM] unexpected error on model={model}, attempt={attempt}/{retries}: {type(e).__name__}: {e}")
            time.sleep(backoff_base)
    raise RuntimeError(f"LLM 호출 실패 (model={model}) after {retries} attempts. last_exc={last_exc}")


# --- high-level: ask for JSON and repair if necessary ---
def call_chat_model_json(
    preferred_models: List[str],
    messages: List[Dict[str, str]],
    *,
    json_retry: int = 3,
    overall_model_retries: int = 2,
    **kwargs
) -> Any:
    """
    preferred_models: list of model names to try in order
    messages: initial conversation
    json_retry: attempts to ask model to output ONLY JSON if initial output is not parseable
    overall_model_retries: tries per model
    Returns parsed JSON object (list/dict) or raises RuntimeError.
    """
    last_errors: List[str] = []
    for model in preferred_models:
        for model_attempt in range(1, overall_model_retries + 1):
            try:
                print(f"[LLM] trying model={model} (model_attempt {model_attempt}/{overall_model_retries})")
                text = call_chat_model_text(model, messages, **{**DEFAULT_PARAMS, **kwargs})
                parsed = _try_parse_json_from_text(text)
                if parsed is not None:
                    return parsed

                # build repair conversation: add assistant output then instruct user to output only JSON
                repair_messages = messages + [
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": (
                            "위의 출력은 JSON 형식으로 파싱되지 않습니다. "
                            "오직 유효한 JSON(예시와 같은 포맷)만을 출력하세요. 추가 설명이나 부가 텍스트를 전혀 출력하지 마세요.\n"
                            "예시: [{\"id\":1, \"text\":\"...\"}, {\"id\":2, \"text\":\"...\"}]\n\n"
                            "다음은 이전 응답의 전체 내용입니다. 해당 내용을 기반으로 유효한 JSON만 다시 출력해 주세요."
                        )
                    }
                ]

                # iterative repair attempts
                for r in range(1, json_retry + 1):
                    print(f"[LLM] JSON repair attempt {r}/{json_retry} on model={model}")
                    text2 = call_chat_model_text(model, repair_messages, **{**DEFAULT_PARAMS, **kwargs})
                    parsed2 = _try_parse_json_from_text(text2)
                    if parsed2 is not None:
                        return parsed2
                    # append assistant output and ask again
                    repair_messages.append({"role": "assistant", "content": text2})
                last_errors.append(f"JSON repair failed on model={model} (attempt {model_attempt})")
            except Exception as e:
                last_errors.append(f"Exception for model={model}: {type(e).__name__}: {e}")
                print(f"[LLM] exception while using model={model}: {e}")
                time.sleep(1.0 * model_attempt)
                continue
    raise RuntimeError(f"All LLM attempts failed. Details: {last_errors}")


# End of file
