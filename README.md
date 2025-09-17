# KFL-AQG-Agents Final (OPENAI_API_KEY handled, complex question types)

This package includes:
- Robust LLM client with retries and JSON repair.
- Agents: sentence_agent, question_agent (supports complex question types), critic_agent.
- Config that loads OPENAI_API_KEY from .env (do NOT put real keys into public repos).

Usage:
1. Copy .env.example to .env and put your OPENAI_API_KEY there.
2. python -m venv .venv
3. source .venv/bin/activate
4. pip install -r requirements.txt
5. python main.py
