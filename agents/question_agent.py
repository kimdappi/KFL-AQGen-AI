from typing import List, Dict
import json
from config import QUESTION_AGENT_MODEL, SENTENCE_AGENT_MODEL, CRITIC_AGENT_MODEL
from utils.llm_client import call_chat_model_json

QUESTION_PROMPT_TEMPLATE = """    You are a Korean language assessment writer. Create a variety of question types from the given contexts.
Learner profile: {profile_json}
Contexts (example sentences/dialogues):
{contexts_json}

Task:
- Generate {n} questions. Use mixed types: multiple_choice, cloze (fill-in-the-blank), dialogue_completion (role-play short turn), sentence_rewrite (transform sentence), error_identification (find the incorrect part), short_answer, and multi-step reasoning question that requires 2-step inference.
- For multiple_choice: 4 choices, one correct. Distractors should reflect common errors for Japanese learners.
- For cloze: Provide the sentence with a blank, the correct token(s), and 3 plausible distractor tokens.
- For dialogue_completion: provide prompt for role-play and expected learner response(s).
- For sentence_rewrite: ask for transformation (e.g., polite->casual, active->passive) and expected answer.
- For error_identification: provide a sentence with an error and indicate which token(s) are wrong and why.
- For multi-step: include clear steps needed to solve and the final answer.
- Each question must include: id, type, prompt, choices (if any), answer, explain, difficulty (0.0-1.0)
Output: valid JSON array
"""


def generate_questions(profile: Dict, contexts: List[Dict], n:int=5) -> List[Dict]:
    prompt = QUESTION_PROMPT_TEMPLATE.format(profile_json=json.dumps(profile, ensure_ascii=False), contexts_json=json.dumps(contexts, ensure_ascii=False), n=n)
    messages = [
        {"role":"system", "content": "You write pedagogically sound and varied Korean assessment items."},
        {"role":"user", "content": prompt}
    ]
    preferred = [QUESTION_AGENT_MODEL, SENTENCE_AGENT_MODEL, CRITIC_AGENT_MODEL]
    parsed = call_chat_model_json(preferred, messages, json_retry=4)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ['questions','items','data']:
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
    raise RuntimeError("LLM returned unexpected structure for question generation")
