# main.py
import argparse
import os
from agents.base import UserProfile
from agents.sentence_agent import SentenceAgent, SentenceGenConfig
from agents.critic_agent import CriticAgent, CriticConfig
from agents.question_agent import QuestionAgent, QuestionConfig

def gpt_llm(system: str, user: str) -> str:
    """
    OpenAI Python SDK 1.x 예시 구현
    - 사전 준비: pip install openai
    - macOS/Linux: export OPENAI_API_KEY=YOUR_KEY
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("환경변수 OPENAI_API_KEY가 없습니다. export OPENAI_API_KEY=... 로 설정하세요.")

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # 필요 시 다른 모델로 변경
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--interests", nargs="*", default=["sports", "k-drama"])
    p.add_argument("--nationality", default="JP")
    p.add_argument("--age", type=int, default=20)
    p.add_argument("--gender", default="F")
    p.add_argument("--level", default="beginner", choices=["beginner", "intermediate", "advanced"])
    p.add_argument("--qtype", default="mcq", choices=["mcq", "short", "fill"])
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--templates", default="templates.json")
    return p.parse_args()

def main():
    args = parse_args()
    profile = UserProfile(
        interests=args.interests,
        nationality=args.nationality,
        age=args.age,
        gender=args.gender,
        level=args.level,
    )

    # ① 문장 생성
    sent_agent = SentenceAgent()
    sent_cfg = SentenceGenConfig(n=args.n)
    sentences = sent_agent.generate(profile, sent_cfg).outputs

    # ③ GPT 크리틱(문장)
    critic = CriticAgent(llm=gpt_llm)
    crit_cfg = CriticConfig(use_rewrite=True)
    kept = critic.evaluate_sentences(sentences, profile, crit_cfg).outputs

    # 재시도(옵션)
    if not kept:
        sentences = sent_agent.generate(profile, sent_cfg).outputs
        kept = critic.evaluate_sentences(sentences, profile, crit_cfg).outputs

    # ② 문제 생성(JSON 템플릿)
    q_agent = QuestionAgent(template_path=args.templates)
    q_cfg = QuestionConfig(qtype=args.qtype, difficulty=profile.level)
    items = q_agent.generate(kept, profile, q_cfg).outputs

    # ③ GPT 크리틱(문항)
    final_items = critic.evaluate_items(items, profile, crit_cfg).outputs

    # 저장
    with open("personalized_sentences.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(kept))
    with open("personalized_items.txt", "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(final_items))

    print(f"[DONE] sentences={len(kept)}, items={len(final_items)}")
    print("Saved: personalized_sentences.txt, personalized_items.txt")

if __name__ == "__main__":
    main()
