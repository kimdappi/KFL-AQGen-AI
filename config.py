import os
from dotenv import load_dotenv

# load .env from project root
load_dotenv()

# ====== MODEL CONFIG ======
SENTENCE_AGENT_MODEL = os.getenv("SENTENCE_AGENT_MODEL", "gpt-4o-mini")
QUESTION_AGENT_MODEL = os.getenv("QUESTION_AGENT_MODEL", "gpt-4o-mini")
CRITIC_AGENT_MODEL = os.getenv("CRITIC_AGENT_MODEL", "gpt-4o")

COMMON_PARAMS = {
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 1024
}

# ====== OPENAI KEY (read from env, do NOT hardcode) ======
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 파일에 키를 넣어 주세요.")

# (Optionally) set into os.environ for libraries that read it directly
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# ====== OUTPUT DIR ======
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
