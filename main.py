import json
import os
from typing import Dict, List, Optional, Any
from agents.sentence_agent import generate_sentences
from agents.critic_agent import review_items
from agents.question_agent import generate_questions
from config import OUTPUT_DIR
from dotenv import load_dotenv

load_dotenv()

INPUT_PATH = os.path.join("data", "sample_input.json")


def load_input(path=INPUT_PATH) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_output(filename: str, obj):
    outpath = os.path.join(OUTPUT_DIR, filename)
    with open(outpath, "w", encoding="utf-8") as wf:
        json.dump(obj, wf, ensure_ascii=False, indent=2)
    print("Saved ->", outpath)


def _find_item_index_by_review(items: List[Dict[str, Any]], review: Dict[str, Any]) -> Optional[int]:
    """
    Try to map a critic review element to an item's index in `items`.
    Strategies (in order):
      1) If review['index'] is int and in range -> use (handles 0-based)
      2) If review['index'] is int and 1..len -> treat as 1-based -> use index-1
      3) If review contains 'text' or 'item_text' try to match by content (exact or substring)
      4) If review contains 'question' or similar for question items, try match
      5) Return None if not found
    """
    idx = review.get("index")
    if isinstance(idx, int):
        if 0 <= idx < len(items):
            return idx
        if 1 <= idx <= len(items):
            return idx - 1

    # Common text keys that critic might include
    candidate_texts = []
    for k in ("text", "item_text", "original", "item", "prompt", "question"):
        v = review.get(k)
        if isinstance(v, str) and v.strip():
            candidate_texts.append(v.strip())
        elif isinstance(v, dict):
            # sometimes review may embed the original item
            tv = v.get("text") or v.get("prompt") or v.get("question")
            if isinstance(tv, str) and tv.strip():
                candidate_texts.append(tv.strip())

    # If the critic returns an 'index_str' like "1." or "1)" handle numeric prefix
    idx_str = review.get("index_str") or review.get("index_text")
    if isinstance(idx_str, str):
        s = idx_str.strip()
        # try extracting leading integer
        import re
        m = re.match(r"^(\d+)", s)
        if m:
            num = int(m.group(1))
            if 1 <= num <= len(items):
                return num - 1
            if 0 <= num < len(items):
                return num

    # Try to match by candidate_texts
    for t in candidate_texts:
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                continue
            # try common text fields in item
            it_text = (it.get("text") or it.get("question") or it.get("prompt") or "").strip()
            if not it_text:
                continue
            if it_text == t or it_text in t or t in it_text:
                return i

    # If critic provides an 'original_index' style field
    for key in ("original_index", "item_index", "idx"):
        v = review.get(key)
        if isinstance(v, int):
            if 0 <= v < len(items):
                return v
            if 1 <= v <= len(items):
                return v - 1

    return None


def _process_critic_results(items: List[Dict[str, Any]], crit_results: List[Dict[str, Any]], item_label: str = "item"):
    """
    Generic processing of critic results into passed items.
    Returns a tuple (passed_items_list, warnings_list).
    """
    passed = []
    included_indices = set()
    warnings = []

    for r in crit_results:
        mapped_idx = _find_item_index_by_review(items, r)
        if mapped_idx is None:
            warnings.append(f"[WARN] Critic returned unknown index/reference for {item_label}: {r.get('index')!r}. Full review: {r}")
            continue

        if mapped_idx in included_indices:
            warnings.append(f"[INFO] Duplicate critic review mapping for {item_label} index {mapped_idx}. Skipping duplicate.")
            continue

        if r.get("ok"):
            orig = items[mapped_idx].copy() if isinstance(items[mapped_idx], dict) else items[mapped_idx]
            if r.get("corrected"):
                # Favor explicit 'corrected' field from critic
                if isinstance(orig, dict):
                    # determine appropriate field to update
                    if "text" in orig:
                        orig["text"] = r.get("corrected")
                    elif "question" in orig:
                        orig["question"] = r.get("corrected")
                    else:
                        orig["text"] = r.get("corrected")
                    orig["critic_note"] = r.get("note")
                else:
                    # if item is not dict, skip corrected assignment
                    pass
            passed.append(orig)
            included_indices.add(mapped_idx)
        else:
            warnings.append(f"{item_label.capitalize()} index {mapped_idx} failed critic: {r.get('note')}")

    return passed, warnings


def run_pipeline(input_profile: Dict):
    print("1) Profile:", input_profile)
    topic = input_profile.get("interest", "daily life")
    print("2) Generating example sentences...")
    sentences = generate_sentences(input_profile, topic, n=6)
    print(f"Generated {len(sentences)} sentences/examples.")

    print("3) Critic reviewing sentences...")
    crit_results = review_items(input_profile, sentences)

    # Debug: show critic output structure (short)
    print("DEBUG: critic returned", len(crit_results), "reviews. Sample:", crit_results[:3])

    passed_sentences, warnings = _process_critic_results(sentences, crit_results, item_label="sentence")
    for w in warnings:
        print(w)
    print(f"{len(passed_sentences)} sentences passed critic.")

    if not passed_sentences:
        print("No sentences passed critic. Abort pipeline.")
        return {"ok": False, "reason": "no_sentences_passed"}

    print("4) Generating questions from passed sentences...")
    questions = generate_questions(input_profile, passed_sentences, n=5)
    print(f"Generated {len(questions)} questions.")

    print("5) Critic reviewing questions...")
    crit_q_results = review_items(input_profile, questions)
    print("DEBUG: critic questions returned", len(crit_q_results), "reviews. Sample:", crit_q_results[:3])

    final_questions, q_warnings = _process_critic_results(questions, crit_q_results, item_label="question")
    for w in q_warnings:
        print(w)

    if not final_questions:
        print("No questions passed critic. Abort.")
        return {"ok": False, "reason": "no_questions_passed"}

    problem_set = {
        "profile": input_profile,
        "contexts": passed_sentences,
        "questions": final_questions
    }
    save_output("problem_set.json", problem_set)
    return {"ok": True, "problem_set": problem_set}


if __name__ == "__main__":
    profile = load_input()
    result = run_pipeline(profile)
    print("Pipeline result:", {"ok": result.get("ok")})
