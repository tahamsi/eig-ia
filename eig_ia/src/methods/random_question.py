import random
from typing import Any, Dict

from ..modules.answer_simulator import simulate_answer
from ..modules.hypothesis_scorer import score_hypotheses
from ..modules.oracle_answerers import oracle_answer
from ..modules.question_generator import generate_questions


def run_random_question(dataset: str, example: Dict[str, Any], llm_q, llm_a, llm_scorer, mode: str, k: int) -> Dict[str, Any]:
    observation = example["observation"] if dataset == "art" else example["question"]
    hypotheses = example["hypotheses"] if dataset == "art" else example["rewrites"]
    prior_probs, prior_meta = score_hypotheses(dataset, observation, hypotheses, llm_scorer)
    questions, q_meta = generate_questions(dataset, observation, hypotheses, llm_q, k)
    question = random.choice(questions)
    if mode == "oracle":
        answer = oracle_answer(dataset, question, example)
        a_meta = {"source": "oracle"}
    else:
        answers, a_meta = simulate_answer(dataset, question, llm_a, 1)
        answer = answers[0]
    posterior, s_meta = score_hypotheses(dataset, observation, hypotheses, llm_scorer, question, answer)
    pred = int(posterior.index(max(posterior)))
    return {
        "asked": True,
        "question": question,
        "answer": answer,
        "prior_probs": prior_probs,
        "posterior_probs": posterior,
        "pred": pred,
        "meta": {"question": q_meta, "answer": a_meta, "scorer": s_meta, "prior": prior_meta},
    }
