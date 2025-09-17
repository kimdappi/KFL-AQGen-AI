from typing import List, Dict, Any, Optional
import json
from config import CRITIC_AGENT_MODEL, SENTENCE_AGENT_MODEL, QUESTION_AGENT_MODEL
from utils.llm_client import call_chat_model_json

# Enhanced prompt: explicitly ask for 0-based indices, original_text, and only JSON output.
CRITIC_PROMPT_TEMPLATE = """
You are an expert Korean language teacher and reviewer.
Learner profile: {profile_json}

Task:
- Review the following items (each item is either a sentence or a question) provided in the "Input items" array.
- For each input item, check: grammar, naturalness, level appropriateness, and cultural sensitivity.
- If the item is fine, set ok=true and score an integer 80-100.
- If there is an issue, set ok=false, provide a suggested corrected text (in "corrected"), a short reason in "note", and score an integer 0-79.

REQUIREMENTS (very important):
1) **Return ONLY a valid JSON array** (no extra text, no explanations).
2) Each element of the array must be an object with these keys:
   - "index": integer (0-based index referencing the input items array). **MUST be 0-based.**
   - "ok": boolean
   - "score": integer (0-100)
   - "corrected": string or null (if ok==false and you suggest a correction)
   - "note": short string explaining your decision
   - "original_text": the original item's text (copy from the provided input item) — this helps the system map your review reliably.
3) If you cannot determine an index for an item, include "index": null but still provide "original_text" so the system can match.
4) Provide a small example output (JSON) exactly matching the schema below.

Input items:
{items_json}

Example output (FORMAT EXACTLY; this is just an example):
[
  {{
    "index": 0,
    "ok": true,
    "score": 90,
    "corrected": null,
    "note": "Natural and level-appropriate.",
    "original_text": "이거 한 그릇 주세요."
  }},
  {{
    "index": 1,
    "ok": false,
    "score": 40,
    "corrected": "영수증 주세요.",
    "note": "More natural phrasing for asking for receipt.",
    "original_text": "영수증 부탁합니다."
  }}
]

Now review the items above and output ONLY valid JSON following the schema.
"""

def _normalize_raw_review(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a single raw review object from the LLM into a predictable shape.
    We don't assume index is valid here; mapping to items is done separately.
    """
    r = {}
    # index may be int, str like "1", or missing
    idx = raw.get("index")
    if isinstance(idx, str):
        # try to extract leading integer if present
        try:
            num = int(idx.strip().strip(")."))
            idx = num
        except Exception:
            # leave as-is (may be None or non-numeric)
            try:
                idx = int(raw.get("index_str"))  # sometimes LLM returns index_str
            except Exception:
                idx = None
    if not isinstance(idx, int):
        # try other possible numeric fields
        for k in ("original_index", "item_index", "idx"):
            v = raw.get(k)
            if isinstance(v, int):
                idx = v
                break

    r["index"] = idx if isinstance(idx, int) else None

    # ok coercion
    ok = raw.get("ok")
    if isinstance(ok, str):
        ok_lower = ok.strip().lower()
        ok = ok_lower in ("true", "yes", "y", "ok", "1")
    r["ok"] = bool(ok)

    # score coercion
    score = raw.get("score")
    try:
        score = int(score)
    except Exception:
        # fallback: infer from ok
        score = 90 if r["ok"] else 50
    # clamp
    if score < 0:
        score = 0
    if score > 100:
        score = 100
    r["score"] = score

    # corrected
    corrected = raw.get("corrected")
    if corrected is None:
        # some LLMs use "suggested" or "suggestion"
        corrected = raw.get("suggested") or raw.get("suggestion")
    r["corrected"] = corrected if isinstance(corrected, str) else None

    # note
    note = raw.get("note") or raw.get("reason") or ""
    r["note"] = note if isinstance(note, str) else str(note)

    # original_text helper
    original_text = raw.get("original_text") or raw.get("item_text") or raw.get("text")
    r["original_text"] = original_text if isinstance(original_text, str) else None

    return r

def _map_reviews_to_items(reviews: List[Dict[str, Any]], items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Given normalized reviews (may have index None or 1-based index), map them to item indices robustly.
    Returns a list of reviews with a valid 'index' (0-based) where possible. Reviews that cannot be mapped are skipped
    but printed as warnings. This function ensures no IndexError will occur later.
    """
    mapped: List[Dict[str, Any]] = []
    used_indices = set()

    # helper to find item index by text matching
    def find_by_text(text: str) -> Optional[int]:
        if not text:
            return None
        t = text.strip()
        # exact match first
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                continue
            it_text = (it.get("text") or it.get("question") or it.get("prompt") or "").strip()
            if not it_text:
                continue
            if it_text == t:
                return i
        # substring match
        for i, it in enumerate(items):
            it_text = (it.get("text") or it.get("question") or it.get("prompt") or "").strip()
            if it_text and (t in it_text or it_text in t):
                return i
        return None

    for raw in reviews:
        r = _normalize_raw_review(raw)
        idx = r["index"]

        # If index provided, coerce 1-based -> 0-based when necessary
        if idx is not None:
            if idx < 0:
                # invalid negative index -> treat as None
                idx = None
            else:
                # if index equals len(items) but len>idx-1, check 1-based possibility
                if idx >= len(items) and len(items) > 0 and 1 <= idx <= len(items):
                    idx = idx - 1
        # If no valid index, try mapping by original_text
        if idx is None:
            if r["original_text"]:
                found = find_by_text(r["original_text"])
                if found is not None:
                    idx = found

        # As a last resort, if index is still None but the review contains a short note referencing a number like "1." etc.
        if idx is None and isinstance(raw.get("note"), str):
            import re
            m = re.search(r"(?:index|item)\s*(?:=|:)?\s*(\d+)", raw.get("note"))
            if m:
                try:
                    candidate = int(m.group(1))
                    if 0 <= candidate < len(items):
                        idx = candidate
                    elif 1 <= candidate <= len(items):
                        idx = candidate - 1
                except Exception:
                    pass

        if idx is None:
            print(f"[WARN] Critic review could not be mapped to an item (skipping). Review: {json.dumps(raw, ensure_ascii=False)}")
            continue

        if idx in used_indices:
            # duplicate mapping: skip or log
            print(f"[INFO] Duplicate mapping for item index {idx}. Skipping duplicate review: {json.dumps(raw, ensure_ascii=False)}")
            continue

        # set mapped index
        r["index"] = idx
        mapped.append(r)
        used_indices.add(idx)

    return mapped

def review_items(profile: Dict[str, Any], items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Main entry: calls the LLM critic and returns a normalized list of reviews mapped to item indices.
    Each returned dict contains: index (int, 0-based), ok (bool), score (int), corrected (str|null), note (str), original_text (str|null)
    """
    prompt = CRITIC_PROMPT_TEMPLATE.format(
        profile_json=json.dumps(profile, ensure_ascii=False),
        items_json=json.dumps(items, ensure_ascii=False)
    )
    messages = [
        {"role": "system", "content": "You are a careful and concise reviewer for Korean teaching materials."},
        {"role": "user", "content": prompt}
    ]
    preferred = [CRITIC_AGENT_MODEL, QUESTION_AGENT_MODEL, SENTENCE_AGENT_MODEL]
    parsed = call_chat_model_json(preferred, messages, json_retry=3)

    # parsed may be list or dict with a list key
    raw_reviews = None
    if isinstance(parsed, list):
        raw_reviews = parsed
    elif isinstance(parsed, dict):
        for key in ("reviews", "items", "data", "result"):
            if key in parsed and isinstance(parsed[key], list):
                raw_reviews = parsed[key]
                break

    if raw_reviews is None:
        # unexpected structure; print debug and raise to help debugging
        print("[ERROR] Critic returned unexpected structure. Raw output:", parsed)
        raise RuntimeError("LLM returned unexpected structure for critic review")

    # Normalize and map reviews robustly to items
    mapped_reviews = _map_reviews_to_items(raw_reviews, items)

    # If mapping produced no reviews, raise (or return empty list per previous behavior)
    if not mapped_reviews:
        print("[WARN] Critic returned reviews but none could be mapped to input items.")
        return []

    return mapped_reviews
