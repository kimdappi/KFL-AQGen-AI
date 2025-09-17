from typing import List, Dict
import json
from config import SENTENCE_AGENT_MODEL, QUESTION_AGENT_MODEL, CRITIC_AGENT_MODEL
from utils.llm_client import call_chat_model_json

SENTENCE_PROMPT_TEMPLATE = """    You are an expert Korean language instructor and example-sentence writer focused on learners.
Target learner profile: {profile_json}
Topic / focus: {topic}
Instruction:
- Generate {n} short natural Korean sentences or short dialogues useful for teaching the topic.
- Keep sentences appropriate for the declared level.
- Provide a one-line "note" for each sentence explaining usage or pitfalls.
- Output must be valid JSON: a list of objects with keys: id, text, note, estimated_difficulty(0.0-1.0), tags
"""


def generate_sentences(profile: Dict, topic: str, n:int = 5) -> List[Dict]:
    prompt = SENTENCE_PROMPT_TEMPLATE.format(profile_json=json.dumps(profile, ensure_ascii=False), topic=topic, n=n)
    messages = [
        {"role":"system", "content": "You generate short Korean example sentences and short dialogues for language learners."},
        {"role":"user", "content": prompt}
    ]
    preferred = [SENTENCE_AGENT_MODEL, QUESTION_AGENT_MODEL, CRITIC_AGENT_MODEL]
    parsed = call_chat_model_json(preferred, messages, json_retry=3)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ['items','sentences','examples','data']:
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
    raise RuntimeError("LLM returned unexpected structure for sentence generation")
